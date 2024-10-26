try:
    import io
    from io import BytesIO
    import pandas as pd
    from google.cloud import storage

except Exception as e:
    print("Some Modules are missing{}".format(e))

storage_client = storage.Client.from_service_account_json("MumbaiHacks\\BackEnd\\beaming-signal-434018-a6-c32063515399.json")


bucket = storage_client.get_bucket("mumbai-hacks")

filename = "%s/%s" % ('', 'MyFile.csv')
blob = bucket.blob(filename)
blob.upload_from_filename('Inventory_with_SARIMA_Predictions.csv')
