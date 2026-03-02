import pathlib

p = r"c:\Desktop\NeuroTidy\lone-penguin-site\js\main.js"
c = pathlib.Path(p).read_text(encoding="utf-8")
c = c.replace(r"\`", "`")
c = c.replace(r"\${", "${")
pathlib.Path(p).write_text(c, encoding="utf-8")
print("Fixed syntax errors.")
