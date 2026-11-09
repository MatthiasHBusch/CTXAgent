# Vendored Markdown parser

- **markdown-it 15.0.2**, MIT license; see `markdown-it-LICENSE.txt`.
- Upstream: https://github.com/markdown-it/markdown-it
- Version metadata: https://registry.npmjs.org/markdown-it/15.0.2
- Downloaded package: https://registry.npmjs.org/markdown-it/-/markdown-it-15.0.2.tgz
- Copied only `package/dist/browser/markdown-it.umd.min.js` and `package/LICENSE`.
  No installation scripts executed. No runtime CDN or npm installation required.
- Package integrity verified before copying (npm SHA-512):
  `q4IGxMv56jCqT4OCRCADBoDP3LO4MhmTXjFbphHPXs4g3j9Xg5RDnxqN8IF/3vIWEU+VCnUq+7JUg/cfy2E6Qw==`
- Bundle SHA-256:
  `635972b985228e8af9f0143647c68616b7a3bb09f6946e7e4a52e43dcf5e7be5`
- License SHA-256:
  `792c48c5a849a15fdf9e37e8bcf9e6d1dd13b32b46c642a748a0a46a9919d473`

`index.html` also pins the bundle with Subresource Integrity. If updating it,
verify the package integrity/license, update the script hash/version and tests,
and run the offline Markdown and existing chat/UI checks.

Application policy lives in `../markdown.js`: source HTML is disabled; only
explicit HTTP(S)/mailto links are clickable, with no opener/referrer access.
Model-supplied images become explicit links/alt labels rather than silently
loading remote content. Task checkboxes are disabled. Blank lines between
unambiguous explicit pipe-table rows are repaired for presentation only, outside
parser-recognized code ranges; stored source is never modified. Math/LaTeX,
footnotes, raw HTML, code execution and syntax-highlighting are not enabled.
