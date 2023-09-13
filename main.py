"""
1 : Reading the file
2 : extract ip address and error and success logs
3 : save the output in csv/excel file
4: Send email
"""
import re
import pandas as pd
import pprint
# import smtplib
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# from smtplib import SMTPAuthenticationError
# from email.mime.application import MIMEApplication

logfile = open("serverlogs.log","r")


pattern = r"((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"

ip_addrs_lst = []
failed_lst =[]
success_lst = []
for log in logfile:
    ip_add = re.search(pattern,log)
    ip_addrs_lst.append(ip_add.group())
    lst = log.split(" ")
    failed_lst.append(int(lst[-1]))
    success_lst.append(int(lst[-4]))


total_failed = sum(failed_lst)
total_success = sum(success_lst)
ip_addrs_lst.append("Total")
success_lst.append(total_success)
failed_lst.append(total_failed)
df = pd.DataFrame(columns=['IP Address',"Success","Failed"])
df['IP Address'] = ip_addrs_lst
df["Success"] = success_lst
df["Failed"] = failed_lst

output_file_name = "output.csv"
df.to_csv("output.csv",index=False)

pprint.pprint(df)

# smtp_server = "smtp.gmail.com"
# smtp_port = 587
# email_address = "chatgpt4forus@gmail.com"
# email_password = "LittleSpoon@BaddiSpoon"
# recipient_address = "magarsarthak15@gmail.com"

# # Create the MIME object
# msg = MIMEMultipart()
# msg["From"] = email_address
# msg["To"] = recipient_address
# msg["Subject"] = "Your Output CSV File"

# # Attach the CSV file
# with open(output_file_name, "rb") as f:
#     attach = MIMEApplication(f.read(), _subtype="csv")
#     attach.add_header("Content-Disposition", f"attachment; filename={output_file_name}")
#     msg.attach(attach)

# # Send the email
# server = smtplib.SMTP(smtp_server, smtp_port)
# server.starttls()
# server.login(email_address, email_password)
# server.sendmail(email_address, recipient_address, msg.as_string())
# server.quit()

# import io

# def analyze_logs(logfile_content):
#     pattern = r"((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
#     ip_addrs_lst = []
#     failed_lst =[]
#     success_lst = []

#     for log in logfile_content.split("\n"):
#         ip_add = re.search(pattern,log)
#         if ip_add:
#             ip_addrs_lst.append(ip_add.group())
#         lst = log.split(" ")
#         if len(lst) >= 4:  # Make sure we don't go out of range
#             failed_lst.append(int(lst[-1]))
#             success_lst.append(int(lst[-4]))

#     total_failed = sum(failed_lst)
#     total_success = sum(success_lst)
#     ip_addrs_lst.append("Total")
#     success_lst.append(total_success)
#     failed_lst.append(total_failed)
    
#     df = pd.DataFrame(columns=['IP Address', "Success", "Failed"])
#     df['IP Address'] = ip_addrs_lst
#     df["Success"] = success_lst
#     df["Failed"] = failed_lst

#     csv_buffer = io.StringIO()
#     df.to_csv(csv_buffer, index=False)
#     csv_content = csv_buffer.getvalue()
#     return csv_content
