%% Compatibility fallback for local drafting only.
%% Replace this file with the official Springer Nature sn-jnl.cls before submission.
%% Official template: https://www.springernature.com/gp/authors/campaigns/latex-author-support
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sn-jnl}[2026/08/07 local compatibility fallback]
\LoadClass[12pt]{article}
\RequirePackage[margin=1in]{geometry}
\RequirePackage{authblk}
\RequirePackage{ifthen}
\RequirePackage{natbib}
\RequirePackage{url}
\RequirePackage{hyperref}
\RequirePackage{booktabs}
\newcommand{\fnm}[1]{#1}
\newcommand{\sur}[1]{#1}
\newcommand{\orgdiv}[1]{#1}
\newcommand{\orgname}[1]{#1}
\newcommand{\orgaddress}[1]{#1}
\newcommand{\city}[1]{#1}
\newcommand{\country}[1]{#1}
\renewcommand{\email}[1]{\texttt{#1}}
\let\sn@oldauthor\author
\renewcommand{\author}[2][]{\sn@oldauthor[#1]{#2}}
\let\sn@oldaffil\affil
\renewcommand{\affil}[2][]{\sn@oldaffil[#1]{#2}}
\newcommand{\sn@abstract}{}
\renewcommand{\abstract}[1]{\gdef\sn@abstract{#1}}
\newcommand{\sn@keywords}{}
\newcommand{\keywords}[1]{\gdef\sn@keywords{#1}}
\let\sn@oldmaketitle\maketitle
\renewcommand{\maketitle}{%
  \sn@oldmaketitle
  \begin{abstract}\sn@abstract\end{abstract}
  \noindent\textbf{Keywords: }\sn@keywords\par\bigskip
}
\AtBeginDocument{\bibliographystyle{unsrtnat}}
