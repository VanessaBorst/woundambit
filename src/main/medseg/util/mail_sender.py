import functools
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from medseg.util.path_builder import PathBuilder


def send_mail_on_completion(subject='Function completed'):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                send_mail(subject, f"Function '{func.__name__}' completed successfully.")
                return result
            except Exception as e:
                send_mail(f"{subject} (Exception)", f"Function '{func.__name__}' encountered an error: {e}")
                raise

        return wrapper

    return decorator


def send_mail(subject: str, content: str):
    cfg_filename = "mail_config.json"
    cfg_path = PathBuilder().root().add('data').add('mail').add(cfg_filename).build()
    try:
        with open(cfg_path, "r") as file:
            mail_cfg = json.load(file)
    except FileNotFoundError:
        return
        # print(f"Error: mail config not found under {cfg_path}. Please create it with the required information.")
    except json.JSONDecodeError:
        return
        # print(f"Error: Unable to parse mail config under {cfg_path}. Please check its format.")

    sender_email = mail_cfg["sender_email"]
    receiver_email = mail_cfg["receiver_email"]
    app_password = mail_cfg["app_password"]

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(content, "plain"))

    # Send the email
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            print("Email sent successfully!")
    except Exception as e:
        print(f"Error occurred while sending mail: {e}")
