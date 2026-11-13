# MemUse project site

This folder is the GitHub Pages site for MemUse, memory-bridge, and
traj-recorder. It is plain HTML and CSS. It has no build step.

| File | Contents |
| --- | --- |
| `index.html` | Landing page: case studies, method, project workflow, paper, and entry paths. |
| `results.html` | Results explorer: a sortable table that compares every system, a profile (`?view=profile`) that leads with span verdicts as one whole, then charts the delivered and used entries on a shared axis, and conclusions (`?view=conclusions`) that state what holds across the cohort, each with its evidence chart. Every metric opens a detail card with its meaning, related metric, and memory-form breakdown. |
| `data/results.json` | Verified paper cohort and supplementary MemoraX results, with official project links; see [data provenance](data/README.md). |
| `data/delivered_forms.py` | Adds per-form breakdowns to `results.json` from the retained reports. |
| `assets/results-select.mjs` | Keyboard-accessible result selectors, styled to match the page. |
| `assets/results.mjs` | Results UI. Pure filtering and sorting logic lives in `results-core.mjs`. |
| `assets/results-profile.mjs` | Results profile: verdict strip and its magnified start, the entry-share chart, A − B differences, and linked metric highlighting. |
| `assets/results-detail.mjs` | Metric detail cards, with separate values and memory-form breakdowns for both systems in a comparison, and one task-sequence chart. |
| `data/run_statistics.py` | Reads retained reports into per-run and per-task-window measurements for profile means and sample standard deviations. |
| `data/run_outcomes.py` | Adds each run's task outcomes, delivery volume, and counts of unused relevant entries for the Conclusions view. |
| `assets/results-conclusions.mjs` | Conclusions view: computes each conclusion from the rows and draws its scatter, dumbbell, or bar chart, the summary cards, and the per-system value tables. |
| `assets/conclusions.css` | Conclusions layout, chart marks, and the scroll-in drawing of each chart. |
| `assets/profile.css` | Responsive profile layout, comparison charts, and reduced-motion-aware entrance and update animations. |
| `assets/results-radar.mjs` | The Paper page's six-axis radar on the paper's logarithmic scale. |
| `assets/radar.css` | Radar styles for the Paper page, and the categorical hues both pages share. |
| `assets/paper.mjs` | Selectable case-study timelines and profile-axis comparisons. |
| `assets/img/favicon.svg` | Browser tab icon for every page. |
| `concepts.html` | How MemUse works: spans, credit windows, signals, and verdicts. |
| `tutorial.html` | Evaluate a recording; optional instructions for creating one sample run. |
| `build.html` | Choose an extension point and follow its repository tutorial; development and contribution checks. |
| `paper.html` | Study scope, findings, interactive cases and profiles, the paper abstract, and citation. |
| `assets/site.js` | Product and docs headers, left navigation, search, tabs, and pager. Edit `NAV` to add a docs page. |
| `assets/landing.js` | Memory-action playback and selectable delivery assessments. |
| `assets/landing.css` | Full-width landing layout; shares fonts and theme tokens with the docs. |
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

Keep the landing page concise. Its case studies come from the paper. The memory graph
demo explains a memory trace and its assessment; it is a schematic, not a reported
research result. Its entrance animations replay when `landing.js` re-adds
`.is-entering`; every other state is plain CSS keyed on the demo's attributes. Playback pauses on interaction and when off screen. Reduced
motion preferences disable automatic playback; readers can still start it. Keep
full methods and commands in the docs. The Paper page redraws the case studies
as selectable SVG timelines. Keep its milestones and step numbers aligned with
Figure 4 and its caption. Do not infer actions for unmarked steps. Its profile
chart uses only the `paper` cohort in `data/results.json` and the published radar.
The Results profile shows equal-run means and sample standard deviations.
Verdicts form one 100% strip per system; a second strip magnifies its start
linearly, up to a round share just above the largest non-neutral total, so
neutral stays the remainder and both systems share that scale. The six entry
axes share one linear axis ending at a round value above the largest mean + SD,
with ± one SD whiskers. `axis()` in `results-profile.mjs` owns both tops; a
missing or all-zero chart keeps 0–100%. Only the whisker bounds are clipped;
the printed SD is not. Both charts print their ticks. Paper axis comparisons
retain a linear 0–100% scale. Keep the chart scales explicit.
Keep proportional marks free of minimum pixel widths and gaps: they exaggerate
small shares and shift segment boundaries, especially in mobile comparisons.

Give each topic one main home: verdict rules in `concepts.html`, commands in
`tutorial.html`, and research results in `paper.html`. Link to that section when
another page needs it. Keep full integration contracts in the repository guides.

For results changes, run `node --test site/tests/results.test.mjs`. Keep missing
metrics as missing values, and keep profile shares separate from verdict shares.
Switching between Table, Profile, and Conclusions returns to the top and clears
the previous view's anchor. Changes within a view preserve the reading position.
Direct conclusion links scroll to their target after the result data loads.
The table compares systems; the profile explains one configuration. Keep them
different: do not turn the profile back into a grid of every system. Each side
selects its memory system, harness, model, and thinking effort independently.
Choices cascade through recorded configurations. System selectors stay visible;
the Configure disclosure opens harness, model, and effort settings. Table search
and filters are hidden in Profile and do not constrain either selection. Two
configurations of the same memory system may be compared. Comparisons label both systems on every metric,
use A/B markers and distinct bar patterns, and can swap the complete configurations.
Differences are primary A minus comparison B, calculated from unrounded run means
and shown in percentage points. They remain unavailable if either mean is missing;
they do not claim statistical significance or mark a higher value as better.
Reset clears the comparison and returns to the initial configuration.
The overview and detail cards pair Positive with Benefit and Negative with Harm;
`relation()` in `results-core.mjs` owns those pairs and their wording. Acted on
does not pair with Unresolved: established use can have any effect, and unresolved
influence need not establish use at all.
Hover or keyboard focus highlights the related metric and its strip segments, and
explains the link in a guide that stays in view while the profile scrolls. A strip segment opens the same detail card as its verdict.
Entry axes use quiet hues: steel (`--deliver`) for the three delivered axes and
ochre (`--use`) for Acted on; only Benefit and Harm take the outcome colours. Each
reading matches its bar's colour; in a comparison both take the A/B colours.
Verdicts and axes retain their own sections and units (spans or entries), never
merged into one number. In a profile comparison, the centered detail dialog shows
both systems with their own run summaries; the columns stack on mobile.
A profile detail card ends with one chart of its metric over the task sequence:
each 25-task window uses its own spans, with a mean and ± one SD band across runs.
A comparison draws both systems in that one chart, on shared axes; B is dashed.
`axis()` sets its top and a window without spans breaks the line. The marks stay
hidden until the chart first scrolls into view in the card, then grow in task order.
Table verdict and entry-profile cards show one mean ± SD line per memory form.
Their colours match the form breakdown, with a legend alongside the chart;
the crosshair readout shows each form's mean contribution and SD across runs.
Every task chart has a keyboard-accessible, continuous **Smoothing** slider
from 0–100%, drawn as a pill whose track fills to the value. Zero shows the original 25-task windows. Increasing strength
gradually adds weight to nearby windows; farther windows get less weight.
Both views recompute each run's weighted share before calculating mean and
sample SD across runs. Gaps stay missing. Moving the pointer, or tapping, over a
plot shows a crosshair and a readout card for that window: the sampled task range
and each visible curve's mean and SD. `results.mjs` keeps the card inside the
detail card. Dragging preserves the slider, focus, and scroll position. Marks
follow the slider continuously from what is on screen (an exponential approach,
about 70 ms), so fast input never restarts or jolts the motion; when the axis top
changes, labels and gridlines (elements, not a background) slide together. The
entrance plays only on the first scroll reveal.
Hovering a curve, point, or legend item highlights it without changing the axes;
keyboard focus on a legend does the same. Select a point or legend to isolate
that curve and fit the Y-axis to its mean + SD. Selecting it again or choosing
**Show all** restores the full chart. Other legend choices remain available,
and smoothing preserves the selection. A selection rescales the axis over 560 ms as if it stretched:
the focused curve morphs, the others keep their shape, scale from the baseline,
and fade out; tick labels slide to their new heights while the old ones slide away.
**Show all** reverses it, so returning curves start at the focused scale. Other
legend entries recede while one curve is in focus. Reduced motion skips every morph.
Memory-form breakdowns retain their pooled span populations and are labelled
separately from the run means. The table and Paper page retain pooled memory
metrics. Keep every label explanation to one short line; a test
caps their length. Entrance animations play when `results.mjs` adds
`.is-entering` on first load or a view change. Later renders glide table rows to
their new order and profile marks (`data-mark`) from their previous extent, so a
new selection shows what changed. Reduced-motion preferences disable all of them.

## Conclusions view

The Conclusions view states what holds across every memory configuration of the
selected benchmark. The table compares systems and the profile explains one
configuration; conclusions do neither. Table search and filters do not apply.

- Keep only the conclusions that matter most to a reader who selects or builds
  a memory system. Lead with the limits of recall and task pass rate, then show
  why memory form matters and how to improve the transition from retrieval to use.
  `conclusions()` in `results-conclusions.mjs` owns these four.
- Compute every number from `data/results.json`. Do not write a result into the
  text by hand. A conclusion that the rows cannot support is left out, not shown
  as zero.
- Do not relate two measures when one contains the other. Benefit requires use,
  so Acted on and Benefit are not independent. Ratios of marginal span shares
  are not conditional rates. Show the action share directly.
- Pass rates use exact resolved counts; the reported means are rounded.
  A weak relation does not prove that memory has no effect.
- The design conclusion measures new information among unused relevant entries.
  Use joint entry counts, not ratios of profile averages. Only influence `none`
  counts as unused; selection, reference, and unknown influence do not.
  Bars pool entry counts across runs; dots show each run's share. The linear
  axis spans 0–100%. Missing or empty populations omit this conclusion.
  Label the recommendation as a design priority: the data motivate processing
  after retrieval but do not test that intervention.
- The form conclusion names the forms from the data. It shows where positive
  verdicts concentrate; it does not prove that a form causes benefit.
- A conclusion is a correlation, a concentration, or a comparison. Correlations
  use the Spearman rank correlation (`spearman()` in `results-core.mjs`), which
  needs no assumption about scale. `relationStrength()` describes |ρ|: 0.7 strong,
  0.4 moderate, otherwise weak. Zero has no rank relation. These labels are not
  significance tests: runs repeat tasks, and windows share runs.
- Write close to ASD-STE100: short sentences, active voice, and consistent terms.
  Keep one topic in each sentence and one main point in each conclusion.
  Retain necessary research terms. Say what was observed without causal claims.
  Keep numerical results in the charts, summary cards, and statistic blocks.
  The descriptions explain the finding without numbers. Link the source notes
  beside the view's scope statement.
  End each text column with a takeaway below its statistics. Use the same font
  and size as the conclusion heading, with a decorative lightbulb before it.
  Keep benchmark comparisons generic.
- Tone names the kind of relation, not an outcome: steel (`--deliver`) for a
  positive relation, clay (`--accent`) for a negative one, grey for zero or missing
  correlation, ochre (`--use`) for a conclusion that is not a correlation.
- Scatter points are positioned HTML elements, so they keep their size at every
  width; guide lines stretch in an SVG over the same plot. Systems
  keep the Paper page's hues (`radarSeries`): a second configuration of one
  system is a ring. The range of the no-memory runs is a shaded band.
- Entry shares span decades between systems, so they use logarithmic axes. Zero takes a
  lane of its own at the start of the axis, and the caption says so. The share
  chart uses one domain on both axes, so the dashed line is the limit
  where every relevant entry gives a benefit. Pass-rate axes are cut; the caption
  gives the start.
- The summary cards link to the conclusions. A correlation card draws ρ on a
  −1 to +1 meter with zero at the centre.
- Each scatter has a per-system value table, because points are not focusable.
  Legend chips highlight one system on hover or focus and pin it on selection.
  The pointer reads the nearest point.
- The head and cards enter with the view. Each chart draws when it scrolls into
  view: points from left to right, then the guides. Reduced-motion preferences
  show everything at once.

Edit the HTML files, preview the site, and check local links and command examples
before publishing. Keep commands and flags the same as the
project READMEs and `uv run -m injudge --help`. The remaining static figures come from the
paper source (`fig/*.pdf`), exported as transparent RGBA PNGs at 2,400 pixels
wide. Only the full-page white canvas fills are removed from export copies;
the source PDFs, labels, plots, and colored panels remain unchanged. Figures
blend into the light page; dark mode supplies a light surface for label contrast.
