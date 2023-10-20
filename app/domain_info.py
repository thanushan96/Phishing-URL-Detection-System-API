import whois
from datetime import datetime
def domaincreatedate(url):
    try:
        whois_info = whois.whois(url)
        cd = whois_info.get('creation_date')
        if whois_info.get('domain_name') is None:
            return 'No domain information for this URL'
        if isinstance(cd, list):
            return cd[0].strftime("%a, %d %b %Y %H:%M:%S GMT")
        elif isinstance(cd, datetime):
            return cd.strftime("%a, %d %b %Y %H:%M:%S GMT")
        else:
            return 'No creation date information available'
    except Exception as e:
        print(f"Error: {e}")
        return 'No information about Domain creation date'

def domainexpiredate(url):
    try:
        whois_info = whois.whois(url)
        ed = whois_info.get('expiration_date')
        if whois_info.get('domain_name') is None:
            return 'No domain information for this URL'
        if isinstance(ed, list):
            return ed[0].strftime("%a, %d %b %Y %H:%M:%S GMT")
        elif isinstance(ed, datetime):
            return ed.strftime("%a, %d %b %Y %H:%M:%S GMT")
        else:
            return 'No expiration date information available'
    except Exception as e:
        print(f"Error: {e}")
        return 'No information about Domain expiration date'

def ageofdomain1(url):
    try:
        whois_info = whois.whois(url)
        creation_date = whois_info.creation_date
        expiration_date = whois_info.expiration_date

        if creation_date and expiration_date:
            domain_age_days = (expiration_date - creation_date).days
            return max(0, domain_age_days)
        else:
            return -1  # Indicate that date information is missing
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return -1  # Indicate an error occurred

if __name__ == "__main__":
    # Test the functions with a domain URL
    url = "https://www.google.com/"


    creation_date = domaincreatedate(url)
    print(f"Creation Date for {url}: {creation_date}")


    expiration_date = domainexpiredate(url)
    print(f"Expiration Date for {url}: {expiration_date}")


    age_of_domain = ageofdomain1(url)
    if age_of_domain != -1:
        print(f"Age of {url}: {age_of_domain} days")
    else:
        print(f"Age of {url} cannot be determined or there was an error.")
