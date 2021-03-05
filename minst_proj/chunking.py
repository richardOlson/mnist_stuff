# file to try things

import numpy as np 
import random
from tensorflow import keras
import tensorflow






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
    
    indices_list = []
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
        # indices_list will contain a tuple of the begin and the end and the range between the two
        indices_list.append((begin_all_in, end, list(range(begin_all_in, end + 1))))
        
        # moving to the next data window
        start_index_of_data_window = get_next_data_window_index(data_length, 
                                                start_index_of_data_window, data_window_size)
        if not start_index_of_data_window:
            # breaking out if it is false
            break
    
    return indices_list


# getting the data_chunk size
def get_data_chunk_size(data_size:int, chunk_size:float, in_all:float):
    chunked_size = int(data_size * chunk_size)
    in_all =int(chunked_size * in_all)
    original_data_per_chunk = chunked_size - in_all
    # finding the number of chunck estimated to make
    num_chunks_estimate = int((data_size - in_all)/ original_data_per_chunk)
    return original_data_per_chunk, chunked_size, in_all, num_chunks_estimate



# This is the function that will make the data_chunks
def make_data_chunks(data_length:int, all_in_indices_list:list, orginal_data_size:int,
                        chunked_window_size:int, numChunksEstimated:int):
    
    original_portion_window_size = 0
    start_index_for_window = 0
    current_window_pos = None
    list_of_chunk_indexes = []
    
    build_chunks = -1
    
    while build_chunks < 1: 
        # making it so that when build_chunks flage becomes 0 it will 
        # not allow anymore times through the while loop  at its current
        # time.  This is to stop when the data is done
        if build_chunks == 0:
            build_chunks = 1
        
        current_window_pos = start_index_for_window
        # This set is where we will be adding each separate chunk to
        chunk_indexes = set() 
        
        
        # begin and end are left there for possible use later
        for i in range(len(all_in_indices_list)):
            begin, end , index_val = all_in_indices_list[i]
            # using a set to do the adding of the values
            chunk_indexes.update(index_val)
            
        for j in range(current_window_pos, data_length):
            if original_portion_window_size >= orginal_data_size:
                start_index_for_window += original_portion_window_size

                original_portion_window_size = 0

                if orginal_data_size + start_index_for_window >= data_length:
                    # going to do just one more chunk
                    build_chunks = 0
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

            

# This is the function that will be used to get the data but have it so that there is 
# some of the data that is mixed in all of the data
def chunk_shuffle(data, data_chunk_size=None, in_all=None , rand_seed=None):
    """
    This is the function that will get the data as chunks and having some 
    of the data found in each of the chunks.

    param: data:   The data is the data passed into the function. Not a tuple

    param: data_chunk_size:    This is the size of the data chunk that the function will try to return
                        It is not guaranteed to get the exact amount of chunk size depending on the size 
                        of the data that is passed in the function. Data_chunk_size is a percentage or 
                        float that will be passed in.  For example if .8 would mean that each chunk_size                            will be 80% of the total data.

    param: in_all:             This is the parameter that if passed in will have some of the data that is found in                         all of data chunks.  A float is expected as the variable. This float is as                                  percentage .8 means that each of the chunks will have 80% of the data found in each                         of the data_chunks.
                        If not passed in then there will be no amount overlapping between data_chunks

    :Returns:            Will return a list of data_chunks
    """
    if data_chunk_size == None:
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
                                                                                        data_chunk_size,  in_all)
    
    # getting the random data that is spread through all the data chunks
    # will return a list of tuples, where each tuple has the start and the end
    # indices for some of the data that is in all the chunks
    # This function will check if the data is a tuple, if it is then all the data uses
    # the same indices
    in_all_indices_list = get_in_all_chunks_indices(data, in_all_size, num_chunks_estimate, chunked_window_size, rand_seed=rand_seed)

    # making the data chunks
    # need to make the original_data_size
    # chunkStart:int, windowSize:int, all_in_indices_list:list,                             original_data_per_chunk:int
    chunkList = make_data_chunks(data_length, in_all_indices_list, original_data_per_window_size, chunked_window_size,
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




def reshape_data(data, start_index:int, end_index:int ):
    images = None
    labels = None
    if isinstance(data, tuple):
        images, labels = data
    
    # made the slice to pull from the opposite end
    images = images[start_index: end_index].reshape(-1, (28 * 28))
    labels = labels[start_index: end_index]
    
    images = images/255.0
    
    return images, labels



if __name__ == "__main__":
   
    
    # doing the loading of the data
    train, test = load_images()
    print(f"The size of the train set is: ", len(train[0]))
    # trying the chunking and the shuffling of the data
    train_data = chunk_shuffle(data=train, data_chunk_size=.2, in_all=.13 )

    # printing out the length of each of the data
    counter = 1
    for train_images, train_labels in train_data:
        print(f"The length of the  {counter} images is {len(train_images)}")
        counter += 1


    print(f"We made it to the end")

    