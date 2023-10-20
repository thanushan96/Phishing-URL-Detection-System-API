import ssl
import socket

def get_certificate_issued_to(hostname):
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=hostname) as s:
            s.connect((hostname, 443))
            cert = s.getpeercert()
            subject = dict(x[0] for x in cert['subject'])
            issued_to = subject.get('commonName', "No certification Informations")
            return issued_to
    except:
        return "No certification Informations"

def get_certificate_issued_by(hostname):
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=hostname) as s:
            s.connect((hostname, 443))
            cert = s.getpeercert()
            issuer = dict(x[0] for x in cert['issuer'])
            issued_by = issuer.get('commonName', "No certification Information")
            return issued_by
    except:
        return "No certification Information"


if __name__ == "__main__":
    hostname = "www.google.com"


    issued_by = get_certificate_issued_by(hostname)
    print(f"Issued By: {issued_by}")


    issued_to = get_certificate_issued_to(hostname)
    print(f"Issued To: {issued_to}")
