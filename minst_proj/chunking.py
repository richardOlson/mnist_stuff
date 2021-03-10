# file to try things

import numpy as np 
import random
from tensorflow import keras
import tensorflow
import operator






# This is the funtion that will create the bounds from where the  first indices
# within a data window can be.  Then with a random a value will be chosen 
# from the possible data indices
def begin_all_in_window(dataLength, all_in_per_data_window:int, data_window_size:int, 
                        start_index_of_data_window:int, rand_seed=None):
    if rand_seed:
        random.seed(rand_seed)
    # The window end is not included in the window
    windowEnd = start_index_of_data_window + data_window_size
    # This is to make sure that it doesn't overstep the bounds per window
    if windowEnd > dataLength:
        # need to alter the amount of all_in_that can be used
        windowEnd = dataLength 
        if all_in_per_data_window > (windowEnd - start_index_of_data_window ):
            # need to change the size of the all_in_per_data_window
            all_in_per_data_window = windowEnd - start_index_of_data_window 

    end_bound = windowEnd - all_in_per_data_window
    
    choice = random.randint(start_index_of_data_window, end_bound)
   
    return choice, all_in_per_data_window




# This is the function that will get the beginning index of the next data window
# if there is no more data windows will return false
def get_next_data_window_index(dataSize:int, current_begin_window_index:int, data_window_size:int):
    new_index = current_begin_window_index + data_window_size
    if new_index >= dataSize: # or new_index + data_window_size >= dataSize:
        return False
    return new_index
    




# This is the function that will return the indices of the data
# that is in all of the chunks of data.
def get_in_all_chunks_indices(data, all_in_size:int, num_chunks_estimate:int, chunk_size:int, rand_seed=None):
    
    indices_set = set()
    data_length = None
    # checking to see if the data is a tuple
    if isinstance(data, tuple):
        # Will only look at one but the indices can be used
        # for both data and the data_lables
        data_length = len(data[0])
    else:
        data_length = len(data)
    # Will go through the data by quarters
    # from each quarter will grab 2 1/8th of the all_in size
    data_window_size = int(data_length /8)
    # getting size of 1/8th of the all_in_size
    all_in_per_data_window = int(all_in_size/8)
    start_index_of_data_window = 0
    # doing the loop that will get the indices
    while True:
        begin_all_in , all_in_in_the_window = begin_all_in_window(data_length, all_in_per_data_window, data_window_size, 
                                            start_index_of_data_window, rand_seed=rand_seed)
        end = begin_all_in + all_in_in_the_window
        # this is a set that will hold the in_all_indices
        indices_set.update(list(range(begin_all_in, end + 1)))
        
        # moving to the next data window
        start_index_of_data_window = get_next_data_window_index(data_length, 
                                                start_index_of_data_window, data_window_size)
        if not start_index_of_data_window:
            # breaking out if it is false
            break
    
    return indices_set




# getting the data_chunk size
def get_data_chunk_size(data_size:int, chunk_size:float, in_all:float):
    chunked_size = int(data_size * chunk_size)
    in_all =int(chunked_size * in_all)
    original_data_per_chunk = chunked_size - in_all
    # finding the number of chunck estimated to make
    num_chunks_estimate = int((data_size - in_all)/ original_data_per_chunk)
    return original_data_per_chunk, chunked_size, in_all, num_chunks_estimate




# This is the function that will make the data_chunks
def make_data_chunks(data_length:int, all_in_indices_set:set, orginal_data_size:int,
                        chunked_window_size:int, numChunksEstimated:int):
    
    continueMaking = True
    number_chunks_made = 0
    original_portion_window_size = 0
    start_index_for_window = 0
    current_window_pos = None
    list_of_chunk_indexes = []
    
    build_chunks = -1

    while continueMaking:
        # making it so that when build_chunks flage becomes 0 it will 
        # not allow anymore times through the while loop  at its current
        # time.  This is to stop when the data is done
        if build_chunks == 0:
            build_chunks = 1
        
        current_window_pos = start_index_for_window
        # This set is where we will be adding each separate chunk to
        chunk_indexes = set() 
        
        
        chunk_indexes.update(all_in_indices_set)

        for j in range(current_window_pos, data_length):
            if original_portion_window_size >= orginal_data_size:
                start_index_for_window += original_portion_window_size

                original_portion_window_size = 0
                # counting the number of the chunks made
                number_chunks_made +=1

                if number_chunks_made == numChunksEstimated:
                    # if in here will loop through the rest of the data
                    # to use up the left overs
                    for k in range(current_window_pos, data_length):
                        if k not in chunk_indexes:
                            chunk_indexes.add(k)
                    continueMaking = False
                    
                break

            # adding to the chunk
            else:
                if j not in chunk_indexes:
                    chunk_indexes.add(j)
                    original_portion_window_size += 1
                
        list_of_chunk_indexes.append(list(chunk_indexes))
    
    return list_of_chunk_indexes

                    


    # indexList, window_index = makeIndexList(chunkStart= window_index, windowSize=chunked_window_size,                                         all_in_indices_list=all_in_indices_list,                       original_data_per_chunk=orginal_data_size)
    # need to build one of the chunks
def chunk_size(data, data_chunk=None, in_all=None):
    """
    This function will retrun the estimated sizes of the chunks, the amount of original data in the 
    chunk and the amount of data that is in all the chunks.

    THESE VALUES ARE NOT EXACT, PARTIALLY BECAUSE MANY TIMES THAT DATA CANNOT BE DIVIDED INTO
    EQUAL AMOUNTS AMONG EACH OF THE CHUNKS
    

    :data:  The data to be used.

    :data_chunk:   Float -- .2 will mean that you want each chunks original data to be 20 percent of data

    :in_all:    Float -- .2 means that each chunk will have 20 percent of the total size of the chunk will be
        found in all the chunks
    """  
    if isinstance(data, tuple):
        data_length = len(data[0])
    else:
        data_length = len(data)
    
    original_data_per_window_size, chunked_window_size, in_all_size, num_chunks_estimate = get_data_chunk_size(data_length, 
                                                                                        data_chunk,  in_all)

    # estimating the size of the data that is original
    available_original = data_length - in_all_size
    original_data_per_window_size = int(available_original / num_chunks_estimate)
    chunked_window_size = original_data_per_window_size + in_all_size

    print(f"The estimated data_chunk size will be aproximately ---  {chunked_window_size}")
    print(f"The estimated size of the original data per data_chunk (window) ---  {original_data_per_window_size}")
    print(f"The estimated size of data that is found in all (in_all) the data chunks --- {in_all_size}")
    print(f"The estimated number of trunks made will be --- {num_chunks_estimate}")





# This is the function that will be used to get the data but have it so that there is 
# some of the data that is mixed in all of the data
def chunk_shuffle(data, data_chunk=None, in_all=None , rand_seed=None):
    """
    This is the function that will get the data as chunks and having some 
    of the data found in each of the chunks.

    param: data:   The data is the data passed into the function. Not a tuple

    param: data_chunk:    This is the size of the data chunk that the function will try to return
                        It is not guaranteed to get the exact amount of chunk size depending on the size 
                        of the data that is passed in the function. Data_chunk is a percentage or 
                        float that will be passed in.  For example if .8 would mean that each chunk_size                            will be 80% of the total data.

    param: in_all:             This is the parameter that if passed in will have some of the data that is found in                         all of data chunks.  A float is expected as the variable. This float is as                                  percentage .8 means that each of the chunks will have 80% of the data found in each                         of the data_chunks.
                        If not passed in then there will be no amount overlapping between data_chunks

    :Returns:            Will return a list of data_chunks
    """
    if data_chunk == None:
        raise Exception("You need to pass in a float value for the data_chunk size")

    data_length = None
    # this is the list that will be returned with the data
    # if there is a x and a y value then the list will contain a list of tuples with the tuple
    # being (x, y)
    chunked_data_list = [] 

    if isinstance(data, tuple):
        # will need to pass in to the get_data_chunk_size not a tuple
        data_length = len(data[0])
     
    else:
        data_length = len(data)
    # getting the sizes used in the making of the chunks
    original_data_per_window_size, chunked_window_size, in_all_size, num_chunks_estimate = get_data_chunk_size(data_length, 
                                                                                        data_chunk,  in_all)
    
    # getting the random data that is spread through all the data chunks
    # will return a list of tuples, where each tuple has the start and the end
    # indices for some of the data that is in all the chunks
    # This function will check if the data is a tuple, if it is then all the data uses
    # the same indices
    in_all_indices_set = get_in_all_chunks_indices(data, in_all_size, num_chunks_estimate, chunked_window_size, rand_seed=rand_seed)
    # Finding the new size of the or
    # making the data chunks
    # need to make the original_data_size
    size_in_all = len(in_all_indices_set)
    total_amount_data = data_length - size_in_all
    
    original_data_per_window_size = int(total_amount_data / num_chunks_estimate)

    print("\nThese are the real values for the sizes for the data chunks")
    print(f"The size of data in all chunks is {size_in_all}")
    print(f"The size of the original data in each chunk is {original_data_per_window_size}")
    # chunkStart:int, windowSize:int, all_in_indices_list:list,                             original_data_per_chunk:int
    chunkList = make_data_chunks(data_length, in_all_indices_set, original_data_per_window_size, chunked_window_size,
                    num_chunks_estimate)
    
    # will then make the data by using the list for each chunk
    
        # if it is a tuple will assume that one in the x and the other is the y
    # looping through the chunklist indexes
    for chunk in chunkList:
        if isinstance(data, tuple):
            x = data[0][chunk]
            y = data[1][chunk]
            chunked_data_list.append((x,y))
        else:
            x = data[chunk]
            chunked_data_list.append(x)

    return chunked_data_list



# getting the data for another block  that is different from the one that the modle above is trained with.
def load_images():
    training, testing = keras.datasets.mnist.load_data()
    # Each of these are tuples that contain ndarrays

    return training, testing




def reshape_data(data, start_index:int, end_index:int, shape:tuple, rbg_val=255.0 ):
    images = None
    labels = None
    if isinstance(data, tuple):
        images, labels = data
    
    # made the slice to pull from the opposite end
    images = images[start_index: end_index].reshape(-1, shape[0] * shape[1])
    labels = labels[start_index: end_index]
    
    images = images/rbg_val
    
    return images, labels





# below are the functions for finding the weights on the average and also finding the loss

# This is the function that will give the loss or the accuracy in a list
def get_loss_or_acc(historyList:list, loss=None, acc=None):
    """
    This function will return the history or the loss of each of the models.

    param:  Loss should be  the type of loss (string) that is found in the dictionary of history
            Acc if not None should be a string of the name that person wants to get from the history
            can only have either the loss or the acc passed in.

    Returns:    Will return a list of the loss or the acc in the order that the histories are passed in.
                Returns the last value in the history list.
    """
    if loss == None and acc == None:
        raise Exception ("Need to have at either loss or acc not be None")
    value_list = []
    item  = loss
    if acc:
        item = acc
    
    for h in historyList:
        value_list.append(h.history[item][-1])
    return value_list



# This function will find the average using the loss or the acc 
# If using the loss the best one is the one with the least loss 
# If using the acc then the best one is the one with the highest acc
def get_avg_with_metric(listArr:list, amount:float, loss=None, acc=None ):
    best_val = None
    best_arr = 0
    metric = None

    # this is the number to divide by to get the average
    divide_for_avg = 0
    # array that will hold all the values and will hold the end result of the avg 
    # array
    avg_arr = np.zeros(shape=listArr[0].shape)

    # the amount will be if you want it to be by the tenth, hundreth or the thousandth
    # for example .1 is tenth, .01 hundreth, .001 thousandth
    multiplier =int(1 / amount)
    # this is used to do the number of loops for adding each array exepct the best array
    loop_num = 0
    
    if loss != None:
        # loss must be a list
        # need to find the lowest loss
        comp = operator.lt
        metric = loss
        # setting to a high nunber for the loss to
        # be able to find something that is lower than this
        best_val = 1000 
    else:
        comp = operator.gt
        metric = acc
        best_val = 0
    # doing the looping find the array that is the best
    for i, val in enumerate(metric):
        if comp(val, best_val):
            best_val = val
            best_arr = i
    # adding the correct amount to each of the array
    for i, arr in enumerate(listArr):
        if i == best_arr:
            # doing the best one into the avg_arr
            divide_for_avg += multiplier
            for _ in range(multiplier):
                avg_arr += arr
        else:
            if loss != None:
                # doing a loss
                loop_num = round(((best_val/loss[i]) * multiplier))
            else:
                loop_num = round(((acc[i]/best_val) * multiplier))
            divide_for_avg += loop_num
            # doing the looping and adding the array value to a
            for _ in range(loop_num):
                avg_arr += arr
    # will now divide by the number to get the average
    avg_arr = avg_arr/divide_for_avg
    return avg_arr
    



# This makes a list of the numpy array at the correct levl
def  makeList(allWeights, level:int):
    theList = []
    for i in range(len(allWeights)):
        theList.append(allWeights[i][level])
    return theList



def create_weight_avg(allWeights:list, loss=None, acc=None, amount=None):
    """
    Function to create a average of the weights.

    If we want to make the averages based on the loss we put a list of the losses 
    which will correspond to the weights.  If we want it based on the accuracy, 
    then we will put in a list of the accuracies for each of the weights.

    Amount is used when doing loss or accuracy.  It is the amount of accuracy or loss precision.
    can be .1, .01, .001 ect.

    Returns:  will return the new list of the weights which can then be used to set the weights 
        of the model.
    """
    # list of the numpy
    numpyList = []
    
    
    # doing a looping through the weights from each model
    for i in range(len(allWeights[0])):
        # making the val a numpy array used for holding the average
        val = np.zeros(allWeights[0][i].shape)
        # outher loop doing the number of the numpy arrays in each list  in the list
        if loss != None or acc != None:
            # makeList will make a list of all the numpy values from each 
            # of the models passed in at a certain level
            npList = makeList(allWeights, level=i)  
            # calling the function to get the avg val of the list of numpy values
            val = get_avg_with_metric(loss=loss, acc=acc, listArr=npList, amount= amount)
        else:
            for j in range(len(allWeights)):
                # if loss != None or acc != None:
                    # # makeList will make a list of all the numpy values from each 
                    # # of the models passed in at a certain level
                    # npList = makeList(allWeights, level=i)  
                    # # calling the function to get the avg val of the list of numpy values
                    # val = get_avg_with_metric(loss=loss, acc=acc, listArr=npList, amount= amount)
               # else:  
               # adding the values of from the numpy arrays to get the total of each of the numpy arrays      
                val += allWeights[j][i]
            # making the average when we are not using the loss or the accuracy
            val = val/len(allWeights)

        # if loss != None or acc != None:        
        #     # finding the average of the weights of one layer
        #     val = val/len(allWeights)

        # putting the val numpy array into the list
        numpyList.append(val)
    return numpyList




if __name__ == "__main__":
   
    
    # doing the loading of the data
    train, test = load_images()
    print(f"The size of the train set is: ", len(train[0]))
    # trying the chunking and the shuffling of the data

    print(f"This is using the size function to see the size before being used")
    chunk_size(train, data_chunk=.15, in_all=.13)

    print("\nNow we are doing the chunk_shuffle and will look at the sizes")
    train_data = chunk_shuffle(data=train, data_chunk=.15, in_all=.13 )

    # printing out the length of each of the data
    counter = 1
    for train_images, train_labels in train_data:
        print(f"The length of the  {counter} images is {len(train_images)}")
        counter += 1


    print(f"We made it to the end")

    