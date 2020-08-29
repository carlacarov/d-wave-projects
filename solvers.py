from dwave.cloud import Client
#read txt file with my token
def loadtoken():
    f= open("mytoken.txt","r")
    return f.read()

client = Client.from_config(token=loadtoken())
solvers = client.get_solvers()
print (solvers)