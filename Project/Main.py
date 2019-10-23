from Tests.orginal_u_net import orginal_u_net
from Tests.orginal_u_net_LSTM import orginal_u_net_lstm

def main(argv):
   
    if argv == '1':
        print('doing orginal_u_net')
        history_1 = orginal_u_net()
    elif argv == '2':
        print('doing orginal_u_net_lstm')
        history_2 = orginal_u_net_lstm()  
    else:
        print('wrong task number')


if __name__ == "__main__":
    #input in the console is the number of the task
    task = input("Enter the number of task to perform: ")
    main(task)
    

