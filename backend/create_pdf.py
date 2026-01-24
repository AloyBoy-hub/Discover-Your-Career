from reportlab.pdfgen import canvas

def create_resume_pdf(filename):
    c = canvas.Canvas(filename)
    c.drawString(100, 800, "John Doe")
    c.drawString(100, 780, "Email: john.doe@example.com")
    c.drawString(100, 760, "Phone: +65 9123 4567")
    
    c.drawString(100, 700, "Skills:")
    c.drawString(120, 680, "- Python")
    c.drawString(120, 660, "- Docker")
    c.drawString(120, 640, "- AWS Cloud")
    c.drawString(120, 620, "- React.js")
    
    c.drawString(100, 580, "Experience:")
    c.drawString(100, 560, "Software Engineer at Tech Corp")
    
    c.save()

if __name__ == "__main__":
    create_resume_pdf("sample_resume.pdf")
