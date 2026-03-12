import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading

# CONFIGURATION - REPLACE WITH YOUR CREDENTIALS
# Google App Password is required for Gmail
SENDER_EMAIL = "dharanish5624@gmail.com" 
SENDER_PASSWORD = "fglz brvh ouic jpzn" 

def send_email_thread(to_email, subject, body):
    if not to_email or "@" not in to_email:
        print("[EMAIL] Invalid email address, skipping.")
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        # Connect to Gmail SMTP Server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, to_email, text)
        server.quit()
        print(f"[EMAIL] Sent to {to_email}")
    except Exception as e:
        print(f"[EMAIL] Failed to send: {e}")

def send_attendance_email(to_email, name, status, time_str):
    subject = f"Attendance Alert: {name} - {status}"
    body = f"""
    Dear {name},

    This is a notification from the Smart Attendance System.

    Status: {status}
    Time: {time_str}

    If this was not you, please contact the administrator.

    Regards,
    Admin
    """
    # Run in background thread to not block camera
    threading.Thread(target=send_email_thread, args=(to_email, subject, body)).start()

def send_registration_email(to_email, name, roll):
    subject = f"Welcome to Smart Attendance: {name}"
    body = f"""
    Dear {name},

    Welcome! You have been successfully registered in the Smart Attendance System.

    Roll Number: {roll}
    Registered Email: {to_email}

    You can now use the 'Take Attendance' feature to mark your presence.

    Regards,
    Admin
    """
    threading.Thread(target=send_email_thread, args=(to_email, subject, body)).start()
