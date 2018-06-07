from models import server

test_input = {'img': '', 'pclds': ''}
server_ins = server.Server()
server_ins.predict(test_input)
