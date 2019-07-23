def write_latex_header(f2):
    f2.write(r'\documentclass[10pt]{article}'+'\n')
    f2.write(r'\usepackage{graphicx}'+'\n')
    f2.write(r'\usepackage{subfig}'+'\n')
    f2.write(r'\usepackage[left=0.25in, right=0.25in, top=0.25cm]{geometry}'+'\n')
    f2.write(r'\begin{document}'+'\n')
    
    f2.write(r'\begin{figure}'+'\n')
    f2.write(r'\centering'+'\n')
    
    return f2

def write_latex_footer(f2):
    f2.write(r'\end{figure}'+'\n')
    f2.write(r'\end{document}')

    return f2
