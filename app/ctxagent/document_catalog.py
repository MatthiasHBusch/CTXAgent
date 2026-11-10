"""Late-bound document operations; one stable native schema, a mutable catalog.

Catalog messages go with the live views (behind the newest cache rung) and before live file data. They
are generated, not persisted as history. Open/close changes never change the
native dispatcher schema; provider cache hits remain an external measurement.
"""
import copy
import json

from . import document_edits, tokens

DISPATCHER = 'document_call'


def _canvas_operations():
    from .canvas import OPERATIONS
    return frozenset(OPERATIONS)


#: Operations that exist only while something compatible is open: per-document edits
#: (a ready document view) and canvas operations (an open canvas).
LATE_BOUND = frozenset(document_edits.TOOLS) | _canvas_operations()


def available(name, conv) -> bool:
    """No I/O: whether `name` can run on what `conv` has open right now."""
    if name in _canvas_operations():
        return conv.canvas.is_open
    return document_edits.available(name, conv.workspace)


def operation_names(session):
    return [n for n in session.tool_names() if n in LATE_BOUND and available(n, session.conv)]


def enabled(session):
    return any(n in LATE_BOUND for n in session.cfg.enabled_tools)


def parameters(operation):
    """Publish and validate the same closed object schemas, including nested edits."""
    from .tools import REGISTRY
    schema = copy.deepcopy(REGISTRY[operation].parameters)
    def close(node):
        if isinstance(node, dict):
            if node.get('type') == 'object':
                node.setdefault('additionalProperties', False)
            for value in node.values():
                close(value)
        elif isinstance(node, list):
            for value in node:
                close(value)
    close(schema)
    return schema


def messages(session):
    from .tools import REGISTRY
    from . import dynamic_tools
    names = operation_names(session)
    if not names or dynamic_tools.active(session):
        return []  # native deferred tools carry the operations instead
    payload = {
        'operations': [{'name': n, 'description': REGISTRY[n].description,
                        'parameters': parameters(n)} for n in names],
        'documents': [{'id': v['id'], 'operations': [n for n in names if n in
                       document_edits.tools_for_path(v['path'])]}
                      for v in session.conv.workspace.views if v.get('status') == 'ready']
                     + ([{'id': 'canvas', 'operations': [n for n in names if n in _canvas_operations()]}]
                        if session.conv.canvas.is_open else []),
    }
    return [{'role': 'user', 'content':
        'DOCUMENT OPERATION CATALOG (generated, not file data): operations that '
        'document_call(operation, arguments_json) runs on the open documents. '
        'arguments_json encodes an object matching the parameters.\n'
        + json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}]


def token_count(session):
    found = messages(session)
    return tokens.count_messages(found) if found else 0


def decode(session, operation, arguments_json):
    from jsonschema import Draft202012Validator
    if operation not in operation_names(session):
        raise ValueError('Document operation is not enabled or has no compatible ready open document.')
    if not isinstance(arguments_json, str) or len(arguments_json) > 200000:
        raise ValueError('arguments_json must be JSON text of at most 200000 characters.')
    args = json.loads(arguments_json, object_pairs_hook=document_edits._unique_object,
                      parse_constant=document_edits._reject_constant)
    if not isinstance(args, dict):
        raise ValueError('arguments_json must encode an object.')
    Draft202012Validator(parameters(operation)).validate(args)
    return operation, args
