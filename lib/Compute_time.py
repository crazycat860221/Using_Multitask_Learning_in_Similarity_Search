def Compute_time(start, end, mess):
    time_taken = end - start # time_taken is in seconds
    # hours, rest = divmod(time_taken, 3600)
    # minutes, seconds = divmod(rest, 60)

    # print( '{}: Took {} hours {} minutes {} seconds.'\
    #                 .format(mess, hours, minutes, seconds) )
    
    print( '{}: Took {} seconds.'.format(mess, time_taken) )
        