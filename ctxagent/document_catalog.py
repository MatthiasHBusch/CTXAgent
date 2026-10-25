"""Late-bound document operations; one stable native schema, a mutable catalog.

Catalog messages belong after the frozen prefix and before live file data. They
are generated, not persisted as history. Open/close changes never change the
native dispatcher schema; provider cache hits remain an external measurement.
"""
import copy
import json

from . import document_edits, tokens

DISPATCHER = 'document_call'


def operation_names(session):
    return [n for n in session.tool_names() if n in document_edits.TOOLS
            and document_edits.available(n, session.conv.workspace)]


def enabled(session):
    return any(n in document_edits.TOOLS for n in session.cfg.enabled_tools)


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
    names = operation_names(session)
    if not names:
        return []
    payload = {
        'operations': [{'name': n, 'description': REGISTRY[n].description,
                        'parameters': parameters(n)} for n in names],
        'documents': [{'id': v['id'], 'operations': [n for n in names if n in
                       document_edits.tools_for_path(v['path'])]}
                      for v in session.conv.workspace.views if v.get('status') == 'ready'],
    }
    return [{'role': 'user', 'content':
        'DOCUMENT OPERATION CATALOG — generated runtime capabilities, not file data.\n'
        'Call document_call(operation, arguments_json) using an operation listed below. '
        'arguments_json must encode an object matching its parameters. Open/ready, '
        'format, permission and version guards are rechecked during execution. '
        'Lifecycle tools remain native. Review prepared plans before applying.\n'
        + json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}]


def token_count(session):
    return tokens.count_messages(messages(session)) if operation_names(session) else 0


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
