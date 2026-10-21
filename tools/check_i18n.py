"""Audit the declared UI/diagnostic translation surface without importing Blender.

This is a source contract, not a claim that user prompts, object names or remote
model responses are English. Extend DISPLAY_FUNCTIONS for new status builders.
"""
import ast
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'wondful_ai_renderer'
CJK = re.compile(r'[\u4e00-\u9fff]')
DISPLAY_MODULES = {'ui.py', 'comparison_view.py', 'prompt_editor.py', 'sb_progress.py'}
DISPLAY_FUNCTIONS = {
    'summarize', 'mirrored_note', 'drop_note', 'view_phase', 'block_ref_label',
    'status_text', 'failure_note', 'product_violations', 'env_violations', '_finish_check', '_finish_install', 'summary_line',
}
MESSAGE_VARIABLES = {'msg', 'message', 'note', 'notes', 'warn', 'warning', 'phase'}


def load_catalog():
    spec = importlib.util.spec_from_file_location('wondful_i18n_check', PKG / 'i18n.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def message_id(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out, index = '', 0
        for value in node.values:
            if isinstance(value, ast.Constant):
                out += value.value.replace('{', '{{').replace('}', '}}')
            else:
                out += '{v' + str(index) + '}'
                index += 1
        return out
    return ''


def source_messages(directory=PKG):
    """All literal strings plus normalized f-strings (for stale-key regression)."""
    messages = set()
    for path in directory.glob('*.py'):
        if path.name == 'i18n.py':
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            value = message_id(node)
            if value:
                messages.add(value)
    return messages


def collect_messages(directory=PKG):
    """Yield (context, source, location) from UI, RNA metadata and status sinks."""
    for path in sorted(directory.glob('*.py')):
        if path.name == 'i18n.py':
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        docs = {id(node.body[0].value) for node in ast.walk(tree)
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)}
        for node in ast.walk(tree):
            source = message_id(node)
            if not CJK.search(source) or id(node) in docs or isinstance(parents.get(node), ast.JoinedStr):
                continue
            # Prompt payloads / model instructions are intentionally not localized.
            if '\n' in source or source == '- 当前没有额外 Structure Packet 图片。':
                continue
            context, display = '*', path.name in DISPLAY_MODULES
            cur = node
            while cur in parents:
                cur = parents[cur]
                if isinstance(cur, ast.keyword) and cur.arg in ('name', 'description', 'items', 'message'):
                    display = True
                if isinstance(cur, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    target = ast.unparse(cur.targets[0] if isinstance(cur, ast.Assign) else cur.target)
                    if target in ('bl_label', 'bl_description'):
                        display = True
                        if target == 'bl_label':
                            cls = parents.get(cur)
                            if isinstance(cls, ast.ClassDef) and not any(ast.unparse(b) in ('Panel', 'UIList') for b in cls.bases):
                                context = 'Operator'
                    if target in MESSAGE_VARIABLES or any(s in target for s in
                            ('_phase', '_note', '_message', 'last_error', 'auth_detail', 'prompt_notice', 'model_notice')):
                        display = True
                if isinstance(cur, ast.Raise):
                    display = True
                if isinstance(cur, ast.FunctionDef) and cur.name in DISPLAY_FUNCTIONS:
                    display = True
                if isinstance(cur, ast.Call):
                    func = ast.unparse(cur.func)
                    if func.endswith(('.report', 'notes.append', 'warn.append')) or func in ('_status', '_set_status', '_update_status'):
                        display = True
                if isinstance(cur, ast.Dict):
                    for key, value in zip(cur.keys, cur.values):
                        if isinstance(key, ast.Constant) and key.value in ('skipped', 'reason', 'warning', 'verdict') \
                                and (node is value or node in ast.walk(value)):
                            display = True
            if display:
                yield context, source, f'{path.name}:{node.lineno}'


def audit(entries=None):
    if entries is None:
        entries = load_catalog().TRANSLATIONS['en_US']
    found = {}
    for context, source, location in collect_messages():
        found.setdefault((context, source), location)
    missing = [(context, source, loc) for (context, source), loc in found.items() if (context, source) not in entries]
    return found, missing


def main():
    found, missing = audit()
    print(f'Translation source coverage: {len(found) - len(missing)}/{len(found)} context/message keys')
    for context, source, location in missing:
        print(f'{location} [{context}] {source!r}')
    return bool(missing)


if __name__ == '__main__':
    raise SystemExit(main())
