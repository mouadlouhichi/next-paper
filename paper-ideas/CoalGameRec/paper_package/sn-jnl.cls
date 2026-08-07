%% Compatibility fallback for local drafting only.
%% Replace this file with the official Springer Nature sn-jnl.cls before submission.
%% Official template: https://www.springernature.com/gp/authors/campaigns/latex-author-support
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sn-jnl}[2026/08/07 local compatibility fallback]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[12pt]{article}

\RequirePackage[margin=1in]{geometry}
\RequirePackage{xparse}
\RequirePackage{url}
\RequirePackage{hyperref}
\RequirePackage{booktabs}

\makeatletter
\newcommand{\sn@authors}{}
\newcommand{\sn@affils}{}
\newcommand{\sn@email}{}
\newcommand{\sn@abstract}{}
\newcommand{\sn@keywords}{}

\RenewDocumentCommand{\title}{O{} m}{\gdef\@title{#2}}
\RenewDocumentCommand{\author}{s O{} m}{%
  \ifx\sn@authors\@empty
    \gdef\sn@authors{#3}%
  \else
    \g@addto@macro\sn@authors{, #3}%
  \fi
}
\NewDocumentCommand{\affil}{s O{} m}{%
  \ifx\sn@affils\@empty
    \gdef\sn@affils{#3}%
  \else
    \g@addto@macro\sn@affils{\\#3}%
  \fi
}
\RenewDocumentCommand{\email}{m}{\gdef\sn@email{#1}}
\RenewDocumentCommand{\abstract}{m}{\gdef\sn@abstract{#1}}
\NewDocumentCommand{\keywords}{m}{\gdef\sn@keywords{#1}}

\newcommand{\fnm}[1]{#1}
\newcommand{\sur}[1]{#1}
\newcommand{\orgdiv}[1]{#1}
\newcommand{\orgname}[1]{#1}
\newcommand{\orgaddress}[1]{#1}
\newcommand{\city}[1]{#1}
\newcommand{\country}[1]{#1}

\RenewDocumentCommand{\maketitle}{}{%
  \begin{center}
    {\LARGE\bfseries \@title\par}
    \vspace{1em}
    {\large \sn@authors\par}
    \vspace{0.5em}
    {\small \sn@affils\par}
    \ifx\sn@email\@empty\else\vspace{0.5em}{\small Correspondence: \texttt{\sn@email}\par}\fi
  \end{center}
  \vspace{1em}
  \begin{center}\textbf{Abstract}\end{center}
  \noindent\sn@abstract\par\vspace{1em}
  \noindent\textbf{Keywords: }\sn@keywords\par\bigskip
}
\makeatother

\AtBeginDocument{\bibliographystyle{unsrt}}
