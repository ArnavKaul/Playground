import graphviz
import os

os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'
src = graphviz.Source.from_file("Catalan translation_graph")
src.render("Catalan translation_graph", format="png")