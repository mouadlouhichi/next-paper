#!/bin/bash
# Rebuild the sandbox/local pdfLaTeX verification toolchain in /tmp.
# Usage: bash tools/texlab/bootstrap.sh   (from paper-ideas/ActionShap/code)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p /tmp/texenv /tmp/stubs
cd /tmp/texenv
[ -d node_modules/texlive ] || { npm init -y >/dev/null 2>&1; npm i --no-audit --no-fund texlive >/dev/null 2>&1; }
cd /tmp
[ -d texlive.js-master ] || { curl -sL -o tljs.tar.gz https://codeload.github.com/manuels/texlive.js/tar.gz/refs/heads/master && tar xzf tljs.tar.gz; }
cp "$HERE/threeparttable.sty" /tmp/stubs/
python3 - <<'PYEOF'
stubs={
"setspace":"\\ProvidesPackage{setspace}\n\\newcommand{\\setstretch}[1]{}\n\\newcommand{\\singlespacing}{}\n\\newcommand{\\onehalfspacing}{}\n\\newcommand{\\doublespacing}{}\n\\newenvironment{spacing}[1]{\\begingroup}{\\endgroup}\n\\newenvironment{onehalfspace}{\\begingroup}{\\endgroup}\n\\newenvironment{doublespace}{\\begingroup}{\\endgroup}\n",
"cuted":"\\ProvidesPackage{cuted}\n\\newenvironment{strip}{\\par\\begingroup}{\\endgroup\\par}\n",
"rotating":"\\ProvidesPackage{rotating}\n\\DeclareOption*{}\n\\ProcessOptions*\n\\newenvironment{sidewaysfigure}{\\begin{figure}}{\\end{figure}}\n\\newenvironment{sidewaystable}{\\begin{table}}{\\end{table}}\n\\newcommand{\\rotcaption}[1]{\\caption{#1}}\n",
"wrapfig":"\\ProvidesPackage{wrapfig}\n\\newenvironment{wrapfigure}[2][]{\\begin{figure}}{\\end{figure}}\n",
"appendix":"\\ProvidesPackage{appendix}\n\\DeclareOption*{}\n\\ProcessOptions*\n\\newenvironment{appendices}{\\appendix}{\\par}\n",
"breakurl":"\\ProvidesPackage{breakurl}\n",
"float":"\\ProvidesPackage{float}\n\\makeatletter\n\\let\\as@xfloat\\@xfloat\n\\def\\@xfloat#1[#2]{%\n  \\edef\\as@tmp{\\as@fix{#2}}%\n  \\expandafter\\as@doxf\\expandafter{\\as@tmp}{#1}}\n\\def\\as@doxf#1#2{\\as@xfloat#2[#1]}\n\\def\\as@fix#1{\\as@fixx#1\\@nil}\n\\def\\as@fixx#1#2\\@nil{\\if H#1h!tp\\else#1#2\\fi}\n\\makeatother\n\\newcommand{\\floatstyle}[1]{}\n\\newcommand{\\newfloat}[3]{}\n\\newcommand{\\restylefloat}[1]{}\n",
"placeins":"\\ProvidesPackage{placeins}\n\\newcommand{\\FloatBarrier}{\\par\\clearpage}\n",
"needspace":"\\ProvidesPackage{needspace}\n\\newcommand{\\Needspace}[1]{\\par}\n\\newcommand{\\needspace}[1]{\\par}\n",
"booktabs":"\\ProvidesPackage{booktabs}\n\\providecommand{\\toprule}{\\hline}\n\\providecommand{\\midrule}{\\hline}\n\\providecommand{\\bottomrule}{\\hline}\n\\providecommand{\\cmidrule}[2][]{}\n\\providecommand{\\addlinespace}[1][]{}\n\\providecommand{\\specialrule}[3]{\\hline}\n",
"multirow":"\\ProvidesPackage{multirow}\n\\providecommand{\\multirow}[3]{#3}\n",
"mathrsfs":"\\ProvidesPackage{mathrsfs}\n\\let\\mathscr\\mathcal\n",
"manyfoot":"\\ProvidesPackage{manyfoot}\n\\DeclareOption*{}\n\\ProcessOptions*\n\\providecommand{\\newfootnote}[2][]{}\n",
"xcolor":"\\ProvidesPackage{xcolor}\n\\DeclareOption*{}\n\\ProcessOptions*\n\\providecommand{\\definecolor}[3]{}\n\\providecommand{\\colorlet}[2]{}\n\\def\\color#1{}\n\\providecommand{\\textcolor}[2]{#2}\n\\providecommand{\\colorbox}[2]{#2}\n\\providecommand{\\rowcolor}[2][]{}\n",
"microtype":"\\ProvidesPackage{microtype}\n\\DeclareOption*{}\n\\ProcessOptions*\n",
"vruler":"\\ProvidesPackage{vruler}\n",
"apacite":"\\ProvidesPackage{apacite}\n",
"algorithm":"\\ProvidesPackage{algorithm}\n\\newenvironment{algorithm}[1][]{\\begin{figure}}{\\end{figure}}\n",
"algorithmicx":"\\ProvidesPackage{algorithmicx}\n\\newenvironment{algorithmic}[1][]{\\begingroup}{\\endgroup}\n",
"algpseudocode":"\\ProvidesPackage{algpseudocode}\n\\newcommand{\\Require}{\\par\\noindent\\textbf{Require:}\\ }\n\\newcommand{\\Ensure}{\\par\\noindent\\textbf{Ensure:}\\ }\n\\newcommand{\\State}{\\par\\noindent}\n\\newcommand{\\Statex}{\\par\\noindent}\n\\newcommand{\\For}[1]{\\par\\noindent\\textbf{for} #1 \\textbf{do}\\begingroup}\n\\newcommand{\\EndFor}{\\endgroup\\par}\n\\newcommand{\\ForAll}[1]{\\par\\noindent\\textbf{for all} #1 \\textbf{do}\\begingroup}\n\\newcommand{\\While}[1]{\\par\\noindent\\textbf{while} #1 \\textbf{do}\\begingroup}\n\\newcommand{\\EndWhile}{\\endgroup\\par}\n\\newcommand{\\If}[1]{\\textbf{if} #1 \\textbf{then}\\begingroup}\n\\newcommand{\\Else}{\\endgroup\\par\\textbf{else}\\begingroup}\n\\newcommand{\\EndIf}{\\endgroup\\par}\n\\newcommand{\\Return}{\\par\\noindent\\textbf{return}\\ }\n\\newcommand{\\Call}[2]{#1(#2)}\n",
"tikz":"\\ProvidesPackage{tikz}\n\\newcommand{\\usetikzlibrary}[1]{}\n\\newcommand{\\tikzset}[1]{}\n\\newcommand{\\pgfmathsetmacro}[2]{}\n\\def\\tikz@swallow#1;{}\n\\def\\tikz@opt[#1]{\\tikz@swallow}\n\\def\\tikz@cmd{\\@ifnextchar[\\tikz@opt{\\tikz@swallow}}\n\\let\\node\\tikz@cmd\n\\let\\draw\\tikz@cmd\n\\let\\path\\tikz@cmd\n\\let\\clip\\tikz@cmd\n\\let\\coordinate\\tikz@cmd\n\\long\\def\\foreach#1in#2#3{#3}\n\\newenvironment{tikzpicture}[1][]{\\begingroup}{\\endgroup\\par}\n\\newenvironment{scope}[1][]{\\begingroup}{\\endgroup}\n",
"pdflscape":"\\ProvidesPackage{pdflscape}\n\\newenvironment{landscape}{\\par\\begingroup}{\\endgroup\\par}\n",
}
for k,v in stubs.items():
    open(f"/tmp/stubs/{k}.sty","w").write(v)
print("stubs ready")
PYEOF
D=/tmp/texenv/node_modules/texlive/texlive
cp $D/texmf-dist/ls-R /tmp/lsR.patched
printf './tex/latex/stubs:\n' >> /tmp/lsR.patched
ls /tmp/stubs >> /tmp/lsR.patched
cat > /tmp/shim_head.js <<'EOF'
var self = { postMessage: function (m) { try { var d = JSON.parse(m); if (d && (d.command === "stdout" || d.command === "stderr")) process.stderr.write(d.contents + "\n"); } catch (e) {} } };
EOF
cat > /tmp/driver_tail.js <<'EOF'
if (typeof process !== "undefined" && typeof require === "function" && typeof window === "undefined") {
  var nfs = require("fs"), npath = require("path");
  var cdf = g.Ma.bind(g); var cp = g.oc.bind(g); var failCount = 0;
  function mountDir(host, rel) {
    try{cp("/", rel, true, true);}catch(e2){}
    nfs.readdirSync(host, {withFileTypes:true}).forEach(function(ent){
      var p = npath.join(host, ent.name), r = rel ? rel + "/" + ent.name : ent.name;
      if (ent.isDirectory()) { try{cp("/", r, true, true);}catch(e2){ failCount++; } mountDir(p, r); }
      else if (ent.isFile()) { try{cdf("/" + npath.dirname(r), npath.basename(r), nfs.readFileSync(p), true, true, true);}catch(e2){ failCount++; } }
    });
  }
  (process.env.TL_MOUNTS || "").split(";").forEach(function(spec){
    if (!spec) return;
    if (spec.indexOf(">") !== -1) { var fp = spec.split(">"); var pd = npath.dirname(fp[1]); cdf(pd === "." ? "/" : "/" + pd, npath.basename(fp[1]), nfs.readFileSync(fp[0]), true, true, true); return; }
    var parts = spec.split("="); mountDir(parts[0], parts[1] || "");
  });
  console.log("MOUNT_DONE fails=" + failCount);
  ca = true;
  var rc = e.run(["-interaction=nonstopmode", "-output-format", "pdf", process.env.TL_MAIN || "input.tex"]);
  try { var pdf = g.readFile("/" + (process.env.TL_MAIN || "input.tex").replace(/\.tex$/, ".pdf")); nfs.writeFileSync(process.env.TL_OUT || "/tmp/out.pdf", Buffer.from(pdf)); console.log("DRIVER_OK bytes=" + pdf.length); }
  catch (err) { console.log("DRIVER_NO_PDF err=" + err); }
  process.exit(0);
}
EOF
cat /tmp/shim_head.js /tmp/texlive.js-master/pdftex-worker.js /tmp/driver_tail.js > /tmp/engine_node.js
echo "toolchain ready"
