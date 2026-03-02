import pathlib
import sys

p = r"c:\Desktop\NeuroTidy\lone-penguin-site\js\main.js"
c = pathlib.Path(p).read_text(encoding="utf-8")

# Let's completely rewrite the text formatting block to use simple .split('\n') 
# rather than complex regexes, making it bulletproof against newline weirdness.

old_block = """        // Format bold, code blocks, headers, and lists manually
        text = text.replace(/`([^`]+)`/g, '<code style="color:#2dd4bf; font-weight:bold; background:rgba(0,0,0,0.3); padding:2px 4px; border-radius:4px;">$1</code>');
        text = text.replace(/\\*\\*([^\\*]+)\\*\\*/g, '<strong>$1</strong>');
        text = text.replace(/(?:^|\\n)###\\s+(.*?)(?=\\n|$)/g, '\\n<h4 style="color:#fff; margin-top:25px; margin-bottom:10px; font-weight:700; font-size:14px;">$1</h4>');
        text = text.replace(/(?:^|\\n)##\\s+(.*?)(?=\\n|$)/g, '\\n<h3 style="color:#fff; margin-top:30px; margin-bottom:15px; font-weight:700; font-size:16px;">$1</h3>');
        text = text.replace(/(?:^|\\n)\\*\\s+(.*?)(?=\\n|$)/g, '\\n<div style="margin-left:15px; margin-bottom:8px;"><span style="color:#2dd4bf;">•</span> $1</div>');
        text = text.replace(/(?:^|\\n)-\\s+(.*?)(?=\\n|$)/g, '\\n<div style="margin-left:15px; margin-bottom:8px;"><span style="color:#2dd4bf;">•</span> $1</div>');
        text = text.replace(/\\n/g, '<br>');"""

new_block = """        // Format code blocks and bold globally
        text = text.replace(/`([^`]+)`/g, '<code style="color:#2dd4bf; font-weight:bold; background:rgba(0,0,0,0.3); padding:2px 4px; border-radius:4px;">$1</code>');
        text = text.replace(/\\*\\*([^\\*]+)\\*\\*/g, '<strong>$1</strong>');
        
        // Parse line by line to handle Markdown headings and lists flawlessly
        let formattedLines = text.split('\\n').map(line => {
            let t = line.trim();
            if (t.startsWith('### ')) {
                return '<h4 style="color:#fff; margin-top:25px; margin-bottom:10px; font-weight:700; font-size:14px;">' + t.substring(4) + '</h4>';
            } else if (t.startsWith('## ')) {
                return '<h3 style="color:#fff; margin-top:30px; margin-bottom:15px; font-weight:700; font-size:16px;">' + t.substring(3) + '</h3>';
            } else if (t.startsWith('* ')) {
                return '<div style="margin-left:15px; margin-bottom:8px;"><span style="color:#2dd4bf;">•</span> ' + t.substring(2) + '</div>';
            } else if (t.startsWith('- ')) {
                return '<div style="margin-left:15px; margin-bottom:8px;"><span style="color:#2dd4bf;">•</span> ' + t.substring(2) + '</div>';
            } else {
                return line;
            }
        });
        
        text = formattedLines.join('<br>');"""

if old_block in c:
    c = c.replace(old_block, new_block)
    pathlib.Path(p).write_text(c, encoding="utf-8")
    print("Replaced successfully!")
else:
    print("Could not find old block")
