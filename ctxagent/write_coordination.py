"""Cooperative process-wide serialization of built-in file writes.

OS byte/flock locks are released on process death; no stale PID/TTL stealing.
This does NOT lock arbitrary editors, shell commands or detached jobs. Use disjoint
files/branches for those and hash-checked plans for shared-file changes.
"""
import asyncio
from . import config

MUTATIONS={'write_file','edit_file','apply_edit_plan','document_apply_edits'}


def _try_lock(stream):
    stream.seek(0)
    try:
        if __import__('os').name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        return True
    except (OSError,BlockingIOError):
        return False


def _unlock(stream):
    stream.seek(0)
    if __import__('os').name=='nt':
        import msvcrt
        msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
    else:
        import fcntl
        fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


async def execute(name,args,call):
    if name not in MUTATIONS and not (name=='document_call' and args.get('operation')=='document_apply_edits'):
        return await call()
    config.DATA_DIR.mkdir(parents=True,exist_ok=True)
    with (config.DATA_DIR/'cooperative-write.lock').open('a+b') as stream:
        if stream.tell()==0:
            stream.write(b'\0'); stream.flush()
        while not _try_lock(stream):
            await asyncio.sleep(0.05)
        try:
            task=asyncio.create_task(call())
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # A worker thread may still be writing. Never release early.
                await task
                raise
        finally:
            _unlock(stream)
