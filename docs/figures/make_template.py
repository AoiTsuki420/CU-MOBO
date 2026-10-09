"""Patch pandoc's base reference.docx into a JCIM/ACS-style template:
Times New Roman 12pt, double line spacing. Run from docs/figures/."""
import zipfile, re, shutil, os

base = "base-reference.docx"
out = "jcim-template.docx"

zin = zipfile.ZipFile(base, "r")
items = {n: zin.read(n) for n in zin.namelist()}
zin.close()

# --- styles.xml: docDefaults -> Times New Roman + double spacing ---
s = items["word/styles.xml"].decode("utf-8")
s = s.replace(
    '<w:rFonts w:asciiTheme="minorHAnsi" w:eastAsiaTheme="minorEastAsia" w:hAnsiTheme="minorHAnsi" w:cstheme="minorBidi" />',
    '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman" w:eastAsia="SimSun" />')
# double line spacing in the paragraph default
s = s.replace('<w:spacing w:after="200" />',
              '<w:spacing w:after="120" w:line="480" w:lineRule="auto" />')
items["word/styles.xml"] = s.encode("utf-8")

# --- theme1.xml: force major & minor Latin fonts to Times New Roman ---
tname = "word/theme/theme1.xml"
if tname in items:
    t = items[tname].decode("utf-8")
    t = t.replace('typeface="Calibri Light"', 'typeface="Times New Roman"')
    t = t.replace('typeface="Calibri"', 'typeface="Times New Roman"')
    items[tname] = t.encode("utf-8")

zout = zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED)
for n, data in items.items():
    zout.writestr(n, data)
zout.close()
print("wrote", out, os.path.getsize(out), "bytes")
