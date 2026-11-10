# MemUse project site

This folder is the GitHub Pages site for MemUse, memory-bridge, and
traj-recorder. It is plain HTML and CSS. It has no build step.

| File | Contents |
| --- | --- |
| `index.html` | Orientation: why memory use matters, the three-project workflow, and entry paths. |
| `concepts.html` | How MemUse works: spans, credit windows, signals, and verdicts. |
| `tutorial.html` | Evaluate a recording; optional instructions for creating one sample run. |
| `build.html` | Choose an extension point and follow its repository tutorial; development and contribution checks. |
| `paper.html` | Study scope, findings, figures, the collapsible paper abstract, and citation. |
| `assets/site.js` | Header, left navigation, "On this page", search, tabs, and pager. Edit `NAV` to add a page. |
| `assets/` | Style, script, and paper figures. |
| `.nojekyll` | Tells GitHub Pages to serve the files as they are. |

## Preview locally

```bash
python3 -m http.server 8000 --directory site
```

Open <http://localhost:8000>.

## Publish with GitHub Pages

The workflow `.github/workflows/pages.yml` publishes this folder.

1. Push the `site/` folder and the workflow to the `agentic_search` branch.
2. On GitHub, open the repository. Go to **Settings → Pages**.
3. Under **Build and deployment → Source**, select **GitHub Actions**.
4. Open the **Actions** tab. Select **Deploy site to GitHub Pages**, then
   **Run workflow**. Later pushes that change `site/` publish automatically.
5. When the run is complete, the site is at
   <https://changranxu.github.io/mem-use/>.

Pages availability depends on the repository and account settings. To show the
site link on the repository page, add its URL to the repository’s **About** section.

## Update the site

Give each topic one main home: verdict rules in `concepts.html`, commands in
`tutorial.html`, and research results in `paper.html`. Link to that section when
another page needs it. Keep full integration contracts in the repository guides.

Edit the HTML files, preview the site, and check local links and command examples
before publishing. Keep commands and flags the same as the
project READMEs and `uv run -m injudge --help`. Paper figures come from the
paper source (`fig/*.pdf`), exported as transparent RGBA PNGs at 2,400 pixels
wide. Only the full-page white canvas fills are removed from export copies;
the source PDFs, labels, plots, and colored panels remain unchanged. Figures
blend into the light page; dark mode supplies a light surface for label contrast.
