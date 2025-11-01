"""
Simple Markdown-to-PDF converter using reportlab.
This script reads 'results/ebm_report.md' in the same project and writes 'results/ebm_report.pdf'.
"""

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
MD_PATH = os.path.join(ROOT, 'results', 'ebm_report.md')
OUT_PDF = os.path.join(ROOT, 'results', 'ebm_report.pdf')

if not os.path.exists(MD_PATH):
    raise FileNotFoundError(f"Markdown file not found: {MD_PATH}")

with open(MD_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

# Very small markdown handling: split by double newlines into paragraphs
paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

styles = getSampleStyleSheet()
doc = SimpleDocTemplate(OUT_PDF, pagesize=letter)
story = []

for p in paragraphs:
    # Replace single newlines inside a paragraph with line breaks
    p_html = p.replace('\n', '<br/>')
    story.append(Paragraph(p_html, styles['Normal']))
    story.append(Spacer(1, 8))

# Build PDF
os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
doc.build(story)

print(f"Wrote PDF to: {OUT_PDF}")