LATEX       = pdflatex -interaction=nonstopmode -halt-on-error
BASH        = bash -c
ECHO        = echo
RM          = rm -rf
TMP_SUFFS   = pdf aux bbl blg log dvi ps eps out brf fdb_latexmk fls synctex.gz
CHECK_RERUN =
GIT_PATH    = .git/

NAME = stationary-tags

all: ${NAME}.pdf

${NAME}.pdf: ${NAME}.tex ${NAME}.bib preamble.tex
	${LATEX} ${NAME}
	bibtex ${NAME}
	${LATEX} ${NAME}
	( grep "Rerun to get" ${NAME}.log && ${LATEX} ${NAME} ) || echo "Done."
	( grep "Rerun to get" ${NAME}.log && ${LATEX} ${NAME} ) || echo "Done."


clean:
	${RM} $(foreach suff, ${TMP_SUFFS}, ${NAME}.${suff})
	${RM} *.aux
	${RM} *Notes.bib
