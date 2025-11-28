    with page.expect_download() as download_info:
        target.click(timeout=30000, no_wait_after=True)
    download = download_info.value

    # Persist the file to a known path
    suggested_name = download.suggested_filename
    final_path = r"C:\Users\中務健\Desktop\{}".format(suggested_name)
    download.save_as(final_path)
    print(f"Downloaded -> {final_path}")
