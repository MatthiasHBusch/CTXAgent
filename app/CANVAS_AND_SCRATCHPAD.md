# Canvas, scratchpad and queued messages

Three ways to work with the agent besides the chat: a live image canvas the agent
shows and draws things on, a text scratchpad you both write in, and messages you send
while the agent is still working. Introduced with backend identity **canvas-scratchpad-v1**
(current: **occupancy-autosave-v1**, which adds settings saved on change and the occupancy
panel in provider tokens).

## Messages sent while the agent works

While a turn runs, **Enter queues** the message instead of being refused. Queued
messages appear as chips above the message box; × takes one back into the box until
it has been delivered.

The turn delivers everything queued at the start of its next step: after the
results of the tool batch that was running (and their images), before the next
model request, as ordinary user messages. A final answer with something queued does
not end the turn; the loop continues with the new messages.

A message queued after the last chance to deliver it (the final answer was already
streaming) comes back when the turn ends: after a normal end it is sent as the next
message; after Stop, an error or the iteration limit it goes back into the message
box. A page reload keeps the queue (`GET /api/chats/{id}` lists it).

Endpoints: `POST /api/chats/{id}/queue` (`{message}` or `{scratchpad: true}`; 409 when
no turn is running), `DELETE /api/chats/{id}/queue/{qid}` (404 once delivered).

## Scratchpad

A text field per chat, docked under the chat list in the left sidebar (collapsible,
resizable, at most 20,000 characters). You and the agent both write in it. It is
**not part of the context**: typing changes neither the prompt nor the cache.

- Your typing is saved after 0.5 s and when the field loses focus.
- **Ctrl+Enter** (or the arrow button) hands it to the agent. When the agent is idle
  this starts a turn with the message `[scratchpad] The user asks you to look at the
  shared scratchpad (version N)` and the current text; while it works the same message
  is queued and built with the text current at delivery. The transcript shows it
  folded.
- Agent tools (group "collaboration", enabled by default): `scratchpad_read()` and
  `scratchpad_write(text | append | edits=[{old_string, new_string}])`. Edits are
  exact-match replacements like `edit_file`. The field updates live and flashes when
  the agent writes.
- A save carries the version it started from. If the agent edited in between, its
  appends and edits are re-applied to your text; a full replacement by the agent, or
  an edit whose text you changed, is dropped and the status line says how many.

Stored in the `scratchpads` table of the chat database; deleted with the chat.
Endpoints: `GET` / `PUT /api/chats/{id}/scratchpad` (`PUT {text, base_version}`).

## Canvas

One image per chat that the agent shows to you in the **Canvas** tab of the right
panel and keeps in its own context. The panel switches to the tab when the agent opens
a canvas; later changes put a dot on the tab while you look at another one.

### What the agent has

Native tools:

- `canvas_open(path | width, height, background, label)`: an image file, or a blank
  canvas (default 1600×1000, at most 4096 px per side). Opening another replaces it.
- `view_image(path, canvas=true)`: the same as `canvas_open(path)`.
- `canvas_close()`.

Operations, available only while a canvas is open (through the operation catalog,
or as native deferred tools on the direct Anthropic API, see below):

- `canvas_draw(ops[], label)`: any number of shapes in one call, one version.
  `line`, `polyline`, `arrow`, `rect` (optionally rounded), `ellipse`, `circle`,
  `polygon`, `marker` (dot, cross, x), `text` (size, colour, alignment, wrap width,
  background box, border, padding), `erase` (back to the background), `paste` (an
  image file fitted into a box). Coordinates are canvas pixels. Colours are CSS names
  or `#rrggbb` / `#rrggbbaa`; translucent shapes blend. At most 300 shapes per call;
  an invalid shape rejects the whole call and leaves the canvas unchanged.
- `canvas_script(code, label, timeout)`: Python with Pillow and numpy in its own
  process (default timeout 60 s, at most 300 s). It gets `img` (the current version,
  RGBA), `draw`, `W`, `H`, `font(size)`, the Pillow modules and `np`; whatever `img`
  holds at the end is the next version. `print()` output is returned. It runs with
  your user rights like `run_command`, so it is marked dangerous and read-only
  delegated agents do not get it. matplotlib is not bundled in the installed app.
- `canvas_focus(box | null, rulers)`: zooms the agent's copy to a region (enlarged up
  to 4×) for detail work; the Canvas tab outlines the region. The canvas itself does
  not change.
- `canvas_undo(steps=1 | to_version)`: older versions stay listed, so an undo can be
  reversed with `to_version`. The history keeps the last 60 versions.
- `canvas_save(path, format)`: PNG, JPEG or WebP into a working directory.

### In the context

The current version (or the focused region) is one image at the **end** of every
request: longest side 1024 px, coordinate rulers along two edges (in canvas pixels,
the agent's copy only), and a header with the size, scale, focus and the last
versions. That costs about 1–2k tokens on Claude.

The tail cache mark goes **before** the canvas, so a drawing step re-sends only the
new image; the history before it is read from the cache as usual. The canvas counts
toward context pressure like the other live views.

### In the Canvas tab

The current image (click to enlarge), the focus outline, the version strip (click a
version to look at it; "Show current" returns), **Download PNG** and **Close
canvas**. Closing removes it from the agent's context and discards its versions;
saved files stay. You view the canvas; drawing on it yourself is not part of this
version.

Versions are content-addressed PNG artifacts in the data folder, served by
`/api/chats/{id}/artifacts/{id}` while they belong to the chat's canvas. Endpoints:
`GET /api/chats/{id}/canvas`, `POST /api/chats/{id}/canvas/close` (409 during a turn).

## Native dynamic tools on the direct Anthropic API

With an `ANTHROPIC_API_KEY` and a bare `claude-…` model id that supports
mid-conversation tool changes (Claude Opus 5 / 5.5 / 4.8, Fable 5 / 5.1, Mythos 5 /
5.1, Sonnet 5.5; not Sonnet 5), the late-bound operations (document edits and canvas
operations) are declared in `tools[]` with `defer_loading: true` from the first
request on, beta `mid-conversation-tool-changes-2026-07-01`. There is no
`document_call` and no catalog message; the model calls the operations by name.

When a document or the canvas opens or closes, the history gets a tool-change segment,
sent as a `role: "system"` message of `tool_addition` / `tool_removal` blocks. The
history stays append-only, so neither the cache nor the thinking blocks are lost
by an open or close. The API accepts such a message only after a user turn and
directly before the next assistant turn (or last); a change recorded before more
user content (the canvas image, a message sent meanwhile) is placed behind it.

Before each request the set the history has surfaced is compared with what is open;
the difference is appended. If compaction archived a tool-change segment, the next
request re-adds what is still open. If the API refuses the beta, the chat uses the
catalog for the rest of the process (the request that hit the refusal is resent
with the surfaced operations as ordinary tools). OpenRouter ids (`anthropic/claude-…`)
and every other model keep the catalog; tool-change segments are dropped on those
paths.
