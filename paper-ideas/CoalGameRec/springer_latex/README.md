# Springer Nature LaTeX package for CoalGameRec

This folder contains a Springer Nature compatible LaTeX manuscript source using the journal article template style requested by the user.

## Template source

The official Springer Nature page is:

https://www.springernature.com/gp/authors/campaigns/latex-author-support/see-where-our-services-will-take-you/18782940

The template package linked there is the December 2024 journal article package. It provides `sn-jnl.cls` and bibliography style files.

The sandbox could not download the ZIP because the TLS connection to the CMS resource failed. Therefore, this folder provides `main.tex` written against the official Springer Nature `sn-jnl` class, but it does not vendor the class file.

Before compiling, download the official Springer Nature journal article template ZIP from the page above and copy these files into this folder:

```text
sn-jnl.cls
sn-mathphys-num.bst
sn-basic.bst
sn-chicago.bst
sn-vancouver.bst
```

At minimum, `sn-jnl.cls` and `sn-mathphys-num.bst` are required for the current manuscript.

## Compile

After copying the template files:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Important status

This is not yet a submission-ready manuscript. It is a Springer-formatted draft that incorporates the current LightGCN empirical findings. Before submission, complete:

- registered systematic-review search and PRISMA flow;
- final bibliography;
- ethics determination;
- data and code availability statements;
- external archive for large raw result artifacts;
- all journal declarations;
- final Springer Nature formatting checks.

## Style note

The manuscript avoids em dash characters by request.
