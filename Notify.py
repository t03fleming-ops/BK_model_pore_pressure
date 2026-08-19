import smtplib
from email.message import EmailMessage
from PlotGR import plot_GR


def End_notification(alpha= None, N= None,image = False, content= None):
    with open('.venv/appswrd.txt', 'r') as file:
        pwrd = file.read()

    # Configuration
    sender_email = "t03fleming@gmail.com"
    receiver_email = "t03fleming@gmail.com"  # Sending to yourself
    app_password = pwrd  # Generated from Google Account

    # Create the email
    msg = EmailMessage()
    if content != None:
        msg.set_content(content)
    else:
        msg.set_content("Hello! Your simulation is finished.")
    msg["Subject"] = "Sim Notification"
    msg["From"] = sender_email
    msg["To"] = receiver_email

    if image:
        fname = plot_GR(alpha, N)

        with open(fname, 'rb') as f:
            file_data = f.read()
            file_name = f.name

        msg.add_attachment(
            file_data,
            maintype='application',
            subtype='png',
            filename=file_name
        )

    # Send the email
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error: {e}")