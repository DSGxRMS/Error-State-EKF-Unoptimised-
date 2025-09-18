import fsds, inspect
print("[FSDS module]", fsds.__file__)
from fsds import FSDSClient
client = FSDSClient()
print("[Connecting to]", "127.0.0.1:41451")
client.confirmConnection()
print("Ping to simulator OK")

