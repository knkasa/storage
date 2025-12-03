# downloading.
with page.expect_download() as download_info:
    target.click(timeout=30000, no_wait_after=True)
download = download_info.value

# Persist the file to a known path
suggested_name = download.suggested_filename
final_path = r"C:\Users\中務健\Desktop\{}".format(suggested_name)
download.save_as(final_path)
print(f"Downloaded -> {final_path}")

# Typing texts in search box.  Pressing Tab helps focus the search box.
page.get_by_role("checkbox").nth(1).click()  #open search box
page.wait_for_timeout(500)
page.keyboard.press("Tab")  #press Tab, or try clicking instead 
page.wait_for_timeout(200)
page.keyboard.type("神戸_20251127_テスト", delay=50) #Use keyboard to type

# Either check or click(click is better)
page.locator(".table_header__JobOfferName > div:nth-child(2) > div > div > input").click()
page.locator(".table_header__JobOfferName > div:nth-child(2) > div > div > input").check()

