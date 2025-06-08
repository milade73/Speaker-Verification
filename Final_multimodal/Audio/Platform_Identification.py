def identify_platform(url):
    platforms = {
        "instagram.com": "Instagram",
        "youtu.be": "YouTube",
        "youtube.com": "YouTube",
        "twitter.com": "Twitter",
        "tiktok.com": "TikTok",
        "facebook.com": "Facebook",
        "fb.watch": "Facebook Watch"
    }

    for key in platforms:
        if key in url:
            return f"This link is from {platforms[key]}."

    return "This link is from an unknown or unsupported platform."


# Example usage
link = input("Enter a URL: ").strip()
print(identify_platform(link))
