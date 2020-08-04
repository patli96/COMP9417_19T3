import numpy as np
import csv
import pandas as pd
import time as t
import json
import statistics


def calculate_bluetooth_median():
    """ To calculate the median number of levels of blue tooth.

        We appended all level data of all students in a list, and used the median() function from statistics module.

        Return:
              The median number.
    """
    level_list = []
    for i in range(60):
        number = i
        if i < 10:
            number = "0" + str(i)
        try:
            file_name = f'StudentLife_Dataset/Inputs/sensing/bluetooth/bt_u{number}.csv'
            data = pd.read_csv(file_name, header=0)
            for j in data['level']:
                level_list.append(j)
        except FileNotFoundError:
            continue
    return statistics.median(level_list)


def calculate_wifi_median():
    """ To calculate the median number of levels of wifi.

        We appended all level data of all students in a list, and used the median() function from statistics module.

        Return:
            The median number.
    """
    level_list = []
    for i in range(60):
        number = i
        if i < 10:
            number = "0" + str(i)
        try:
            file_name = f'StudentLife_Dataset/Inputs/sensing/wifi/wifi_u{number}.csv'
            data = pd.read_csv(file_name, header=0)
            for j in data['level']:
                level_list.append(j)
        except FileNotFoundError:
            continue
    return statistics.median(level_list)


def pre_process_flourishing_scale():
    """ To pre-process the flourishing scale data.

        We add the every number up for each student.
        If there's a missing value, we place a mean value of that column.
        If a student has both "pre" and "post" flourishing scale data,
            we take the average as this student's flourishing scale.
        In the end, we calculate a median number of all student's flourishing scale,
            taking it as a threshold.
        If a student's flourishing scale is greater than the threshold, than give a label of 1,
            otherwise, give a label of 0.

        Return:
            result  : A list of pre-processed flourishing scale for each student,
                        in the format of: [ ['u_00', label], ['u_01', label], ...]

    """
    flourishing_dict = {}
    scores_list = []
    mean_score_for_every_column = {}

    with open('StudentLife_Dataset/Outputs/FlourishingScale.csv', mode='r') as file:
        data_file = csv.reader(file)
        flourishing_scale = np.array([rows for rows in data_file])

    # Calculate the mean value for each column.
    for i in range(2, len(flourishing_scale[0])):
        count = 0
        mean_score_for_every_column[i] = 0
        for j in range(1, len(flourishing_scale)):
            if flourishing_scale[j][i]:
                count += 1
                mean_score_for_every_column[i] += int(flourishing_scale[j][i])
        mean_score_for_every_column[i] = mean_score_for_every_column[i] / count

    # Calculate the sum of flourishing scales for each student.
    for i in range(1, len(flourishing_scale)):
        score_sum = 0
        for j in range(2, len(flourishing_scale[i])):
            if flourishing_scale[i][j]:
                score_sum += int(flourishing_scale[i][j])
            else:
                score_sum += mean_score_for_every_column[j]
        scores_list.append(score_sum)
        if flourishing_scale[i][0] in flourishing_dict:
            flourishing_dict[flourishing_scale[i][0]] = (flourishing_dict[flourishing_scale[i][0]] + score_sum) / 2
        else:
            flourishing_dict[flourishing_scale[i][0]] = score_sum

    # Classify the flourishing scale for each student.
    median_flourishing = statistics.median(scores_list)
    for i in flourishing_dict:
        if flourishing_dict[i] <= median_flourishing:
            flourishing_dict[i] = 0
        else:
            flourishing_dict[i] = 1
    result = []
    for student in flourishing_dict:
        result.append([student, flourishing_dict[student]])

    # Store the result in a .csv file and return the result.
    output = pd.DataFrame(result)
    output.to_csv('flourishing.csv', index=False)
    return result


def pre_process_panas():
    """ To pre-process the PANAS data.

        According to the .pdf file given in Outputs file, we recorded the index number belonging to positive PANAS.
        Since there are missing values in some rows, we firstly calculated the mean value of every column
            so that missing values could be filled with the mean value of the columns where they are.
        We add up all the positive PANAS scores and negative PANAS scores respectively,
            if there are both "pre" and "post" type, then we calculate the mean value.
        If the sum is greater than the median number, then a label of 1 will be given.
            Otherwise, a label of 0 will be given.

        Returns:
            positive_result : A list of pre-processed positive PANAS for each student,
                                    in the format of: [ ['u_00', label], ['u_01', label], ...]
            negative_result : A list of pre-processed negative PANAS for each student,
                                    in the format of: [ ['u_00', label], ['u_01', label], ...]
    """
    positive_index = [2, 5, 9, 10, 12, 13, 15, 16, 18]
    positive_dict = {}
    negative_dict = {}
    mean_score_for_every_column = {}
    positive_list = []
    negative_list = []

    with open('StudentLife_Dataset/Outputs/panas.csv', mode='r') as file:
        data_file = csv.reader(file)
        panas = np.array([rows for rows in data_file])

    # Calculate the mean value for each column.
    for i in range(2, len(panas[0])):
        count = 0
        mean_score_for_every_column[i] = 0
        for j in range(1, len(panas)):
            if panas[j][i]:
                count += 1
                mean_score_for_every_column[i] += int(panas[j][i])
        mean_score_for_every_column[i] = mean_score_for_every_column[i] / count

    # Calculate the sum of positive PANAS and negative PANAS for each student.
    for i in range(1, len(panas)):
        positive_sum = 0
        negative_sum = 0
        for j in range(2, len(panas[i])):
            if j in positive_index:
                if panas[i][j]:
                    positive_sum += int(panas[i][j])
                else:
                    positive_sum += mean_score_for_every_column[j]
            else:
                if panas[i][j]:
                    negative_sum += int(panas[i][j])
                else:
                    negative_sum += mean_score_for_every_column[j]
        positive_list.append(positive_sum)
        negative_list.append(negative_sum)
        if panas[i][0] in positive_dict:
            positive_dict[panas[i][0]] = (positive_dict[panas[i][0]] + positive_sum) / 2
        else:
            positive_dict[panas[i][0]] = positive_sum
        if panas[i][0] in negative_dict:
            negative_dict[panas[i][0]] = (negative_dict[panas[i][0]] + positive_sum) / 2
        else:
            negative_dict[panas[i][0]] = positive_sum

    # Calculate the median numbers.
    median_positive_score = statistics.median(positive_list)
    median_negative_score = statistics.median(negative_list)

    # Classify the positive and negative PANAS scores for each student.
    for i in positive_dict:
        if positive_dict[i] <= median_positive_score:
            positive_dict[i] = 0
        else:
            positive_dict[i] = 1
    for i in negative_dict:
        if negative_dict[i] <= median_negative_score:
            negative_dict[i] = 0
        else:
            negative_dict[i] = 1

    # Store the result in two .csv files and return them.
    positive_result = []
    negative_result = []
    for student in positive_dict:
        positive_result.append([student, positive_dict[student]])
    for student in negative_dict:
        negative_result.append([student, negative_dict[student]])
    output_1 = pd.DataFrame(positive_result)
    output_2 = pd.DataFrame(negative_result)
    output_1.to_csv('positive_panas.csv', index=False)
    output_2.to_csv('negative_panas.csv', index=False)
    return positive_result, negative_result


def pre_process(number, bt_median, wf_median, flourishing, positive_panas, neagtive_panas):
    """ To pre-process the data

    This function computes following features (X values):
        1. the activity_duration                2. activity_duration_for_day
        3. activity_duration_for_night          4. stationary_ratio
        5. walking_ratio                        6. running_ratio
        7. silence_ratio                        8. voice_ratio
        9. noise_ratio                          10. conversation_duration
        11. conversation_duration_for_day       12. conversation_duration_for_night
        13. conversation_frequency              14. conversation_frequency_for_day
        15. conversation_frequency_for_night    16. short_conversations
        17. medium_conversations                18. long_conversations
        19. dark_duration                       20. dark_duration_for_day
        21. dark_duration_for_night             22. dark_frequency
        23. dark_frequency_for_day              24. dark_frequency_for_night
        25. locked_duration                     26. locked_duration_for_day
        27. locked_duration_for_night           28. locked_frequency
        29. locked_frequency_for_day            30. locked_frequency_for_night
        31. charged_frequency                   32. traveled_distance
        33. traveled_distance_for_day           34. cell_type_ratio
        35. wifi_type_ratio                     36. gps_type_ratio
        37. sleep_duration                      38. mean_sleep_rate
        39. mean_sleep_tiredness                40. wifi_ratio
        41. bt_ratio                            42. number_of_locations
        43. mean_number_of_ppl_contacted
    And append the target values (Y values) at the end of the X values above:
        1. flourishing_scale
        2. positive_panas
        3. negative_panas

    Store all the features and target values in a list,
        and append the list to the array which stores features of all students named student_features.

    The features (X values) contained in features_according_to_paper are:
        1. the activity_duration                2. activity_duration_for_day
        3. activity_duration_for_night          4. conversation_duration
        5. conversation_duration_for_day        6. conversation_duration_for_night
        7. conversation_frequency               8. conversation_frequency_for_day
        9. conversation_frequency_for_night     10. traveled_distance
        11. traveled_distance_for_day           12. wifi_type_ratio
        13. sleep_duration                      14. mean_number_of_ppl_contacted
    And the three target values (Y values) are also included:
        1. flourishing_scale
        2. positive_panas
        3. negative_panas


    Args:
        number          : The index number of student.
        bt_median       : The median number of blue tooth level.
        wf_median       : The median number of wifi level
        flourishing     : The pre-processed flourishing scales.
        positive_panas  : The pre-processed positive PANAS scores.
        neagtive_panas  : The pre-processed negative PANAS scores.
    """

    global student_features
    global features_according_to_paper

    # Create a list to store all features of current student, and a list to store features mentioned in the paper.
    features = []
    paper_features = []

    # Converse the index number to the expected format.
    if number < 10:
        number = '0' + str(number)
    else:
        number = str(number)

    features.append("u" + number)
    paper_features.append("u" + number)


    # Compute the activity duration, activity duration for day and activity duration for night,
    #   ratio of walking activities and ratio of running activities, i.e.:
    #       1. the activity_duration
    #       2. activity_duration_for_day
    #       3. activity_duration_for_night
    #       4. stationary_ratio
    #       5. walking_ratio
    #       6. running_ratio
    # Store them in a list in the format of:
    #   activity_result = [activity_duration, _for_day, _for_night, stationary_ratio, walking_ratio, running_ratio]
    # Use a try-except statement to check if the file of current student exists.
    # If not, set the list to none.
    try:
        activity = []
        walking_count = 0
        running_count = 0
        stationary_count = 0

        with open(f'StudentLife_Dataset/Inputs/sensing/activity/activity_u{number}.csv') as file:
            data = csv.reader(file)
            allData = np.array([lines for lines in data])
        for i in range(1, len(allData)):
            if len(allData[i]) != 0:
                activity.append(allData[i])

        activity_result = [0, 0, 0, 0, 0, 0]
        active_list = []
        daytime_active_list = []

        for i in range(len(activity)):
            if activity[i][1] != '0':
                active_list.append(activity[i][0])
                # If the start time of the activity is between 6am and 6pm, then we take it as a daytime activity.
                timestamp = int(activity[i][0])
                local_time = t.strftime("%H:%M:%S", t.localtime(timestamp))
                if (local_time >= "06:00:00") and (local_time <= "18:00:00"):
                    daytime_active_list.append(timestamp)
                if activity[i][1] == '1':
                    walking_count += 1
                elif activity[i][1] == '2':
                    running_count += 1
            else:
                stationary_count += 1
                if i == 0 or len(active_list) == 0:
                    pass
                else:
                    activity_result[0] += int(active_list[-1]) - int(active_list[0])
                    active_list.clear()
                    if len(daytime_active_list) != 0:
                        activity_result[1] += daytime_active_list[-1] - daytime_active_list[0]
                        daytime_active_list.clear()

        activity_result[2] = activity_result[0] - activity_result[1]
        activity_result[3] = stationary_count / len(activity)
        activity_result[4] = walking_count / len(activity)
        activity_result[5] = running_count / len(activity)

    except FileNotFoundError:
        activity_result = [None, None, None, None, None]

    for x in activity_result:
        features.append(x)
    for i in range(3):
        paper_features.append(activity_result[i])


    # Compute the ratio of being in an environment of silence, voice and noise, i.e:
    #   7. silence_ratio
    #   8. voice_ratio
    #   9. noise_ratio
    # Use a try-except statement to check if the file of current student exists.
    # If not, set the list to none.
    try:
        with open(f'StudentLife_Dataset/Inputs/sensing/audio/audio_u{number}.csv') as file:
            data = csv.reader(file)
            allData = np.array([lines for lines in data])
        audio = []
        for i in range(1, len(allData)):
            if len(allData[i]) != 0:
                audio.append(allData[i])

        silence_count = 0
        voice_count = 0
        noise_count = 0

        for i in range(len(audio)):
            if audio[i][1] == '0':
                silence_count += 1
            elif audio[i][1] == '1':
                voice_count += 1
            elif audio[i][1] == '2':
                noise_count += 1
        audio_result = [silence_count / len(audio), voice_count / len(audio), noise_count / len(audio)]

    except FileNotFoundError:
        audio_result = [None, None, None]

    for x in audio_result:
        features.append(x)


    # Compute the conversation duration, conversation duration for day, conversation duration for night,
    #   conversation frequency, conversation frequency for day, conversation frequency for night,
    #   number of short conversations, number of medium conversations and number of long conversations, i.e.:
    #       10. conversation_duration
    #       11. conversation_duration_for_day
    #       12. conversation_duration_for_night
    #       13. conversation_frequency
    #       14. conversation_frequency_for_day
    #       15. conversation_frequency_for_night
    #       16. short_conversations
    #       17. medium_conversations
    #       18. long_conversations
    # Store them in three lists in the format of:   conversation_duration = [total_time, for_day, for_night],
    #                                               conversation_frequency = [total_fre, for_day, for_night]
    #                                               conversation_length = [short_, medium_, long_]
    # Use a try-except statement to check if the file of current student exists.
    # If not, set the list to none.
    try:
        with open(f'StudentLife_Dataset/Inputs/sensing/conversation/conversation_u{number}.csv') as file:
            data = csv.reader(file)
            allData = np.array([lines for lines in data])
        conversation = []
        for i in range(1, len(allData)):
            if len(allData[i]) != 0:
                conversation.append(allData[i])

        conversation_duration = [0, 0, 0]
        conversation_frequency = [0, 0, 0]
        conversation_length = [0, 0, 0]

        for i in range(len(conversation)):
            conversation_duration[0] += int(conversation[i][1]) - int(conversation[i][0])
            # If the start time is between 6am and 6pm, then we take it as a daytime conversation.
            timestamp = int(conversation[i][0])
            local_time = t.strftime("%H:%M:%S", t.localtime(timestamp))
            if (local_time >= "06:00:00") and (local_time <= "18:00:00"):
                conversation_duration[1] += int(conversation[i][1]) - int(conversation[i][0])
                conversation_frequency[1] += 1
            timestamp = int(conversation[0][1])
            local_time = t.localtime(timestamp)
            hour_1 = local_time.tm_hour
            min_1 = local_time.tm_min
            sec_1 = local_time.tm_sec
            timestamp = int(conversation[0][0])
            local_time = t.localtime(timestamp)
            hour_2 = local_time.tm_hour
            min_2 = local_time.tm_min
            sec_2 = local_time.tm_sec
            duration_in_minutes = (hour_1 - hour_2) * 60 + (min_1 - min_2) + (sec_1 - sec_2) / 60
            if duration_in_minutes <= 5:
                conversation_length[0] += 1
            elif 5 < duration_in_minutes <= 30:
                conversation_length[1] += 1
            else:
                conversation_length[2] += 1

        conversation_duration[2] = conversation_duration[0] - conversation_duration[1]
        conversation_frequency[0] = len(conversation)
        conversation_frequency[2] = conversation_frequency[0] - conversation_frequency[1]

    except FileNotFoundError:
        conversation_duration = [None, None, None]
        conversation_frequency = [None, None, None]
        conversation_length = [None, None, None]

    for x in conversation_duration:
        features.append(x)
        paper_features.append(x)
    for x in conversation_frequency:
        features.append(x)
        paper_features.append(x)
    for x in conversation_length:
        features.append(x)


    # Compute the dark duration, dark duration for day, dark duration for night,
    #   dark frequency, dark frequency for day, dark frequency for night, i.e.:
    #       19. dark_duration
    #       20. dark_duration_for_day
    #       21. dark_duration_for_night
    #       22. dark_frequency
    #       23. dark_frequency_for_day
    #       24. dark_frequency_for_night
    # Store them in two lists in the format of:   dark_duration = [total_time, for_day, for_night],
    #                                               dark_frequency = [total_fre, for_day, for_night]
    # Use a try-except statement to check if the file of current student exists.
    # If not, set the list to none.
    try:
        with open(f'StudentLife_Dataset/Inputs/sensing/dark/dark_u{number}.csv') as file:
            data = csv.reader(file)
            allData = np.array([lines for lines in data])
        dark = []
        for i in range(1, len(allData)):
            if len(allData[i]) != 0:
                dark.append(allData[i])

        dark_duration = [0, 0, 0]
        dark_frequency = [0, 0, 0]

        for i in range(len(dark)):
            dark_duration[0] += int(dark[i][1]) - int(dark[i][0])
            timestamp = int(dark[i][0])
            local_time = t.strftime("%H:%M:%S", t.localtime(timestamp))
            if (local_time >= "06:00:00") and (local_time <= "23:00:00"):
                dark_frequency[1] += 1
                dark_duration[1] += int(dark[i][1]) - int(dark[i][0])
        dark_duration[2] = dark_duration[0] - dark_duration[1]
        dark_frequency[0] = len(dark)
        dark_frequency[2] = dark_frequency[0] - dark_frequency[1]

    except FileNotFoundError:
        dark_duration = [None, None, None]
        dark_frequency = [None, None, None]

    for x in dark_duration:
        features.append(x)
    for x in dark_frequency:
        features.append(x)


    # Compute the locked duration, locked duration for day, locked duration for night,
    #   locked frequency, locked frequency for day, locked frequency for night, i.e.:
    #       25. locked_duration
    #       26. locked_duration_for_day
    #       27. locked_duration_for_night
    #       28. locked_frequency
    #       29. locked_frequency_for_day
    #       30. locked_frequency_for_night
    # Store them in two lists in the format of:   locked_duration = [total_time, for_day, for_night],
    #                                               locked_frequency = [total_fre, for_day, for_night]
    # Use a try-except statement to check if the file of current student exists.
    # If not, set the list to none.
    try:
        with open(f'StudentLife_Dataset/Inputs/sensing/phonelock/phonelock_u{number}.csv') as file:
            data = csv.reader(file)
            allData = np.array([lines for lines in data])
        locked = []
        for i in range(1, len(allData)):
            if len(allData[i]) != 0:
                locked.append(allData[i])

        locked_duration = [0, 0, 0]
        locked_frequency = [0, 0, 0]

        for i in range(len(locked)):
            locked_duration[0] += int(locked[i][1]) - int(locked[i][0])
            timestamp = int(locked[i][0])
            local_time = t.strftime("%H:%M:%S", t.localtime(timestamp))
            if (local_time >= "06:00:00") and (local_time <= "23:00:00"):
                locked_frequency[1] += 1
                locked_duration[1] += int(locked[i][1]) - int(locked[i][0])
        locked_duration[2] = locked_duration[0] - locked_duration[1]
        locked_frequency[0] = len(locked)
        locked_frequency[2] = locked_frequency[0] - locked_frequency[1]

    except FileNotFoundError:
        locked_duration = [None, None, None]
        locked_frequency = [None, None, None]

    for x in locked_duration:
        features.append(x)
    for x in locked_frequency:
        features.append(x)


    # Compute frequency of the phone being charged, i.e:
    #   31. charged_frequency
    try:
        with open(f'StudentLife_Dataset/Inputs/sensing/phonecharge/phonecharge_u{number}.csv') as file:
            data = csv.reader(file)
            allData = np.array([lines for lines in data])
        charged = []
        for i in range(1, len(allData)):
            if len(allData[i]) != 0:
                charged.append(allData[i])
        features.append(len(charged))

    except FileNotFoundError:
        features.append(None)


    # Compute the traveled distance, and the ratio of indoor, on campus and outdoor activities,
    #   depending on the signal type, i.e.:
    #       32. traveled_distance
    #       33. traveled_distance_for_day
    #       34. cell_type_ratio
    #       35. wifi_type_ratio
    #       36. gps_type_ratio
    # Use a try-except statement to check if the file of current student exists.
    # If not, set the list to none.
    col_name = ['provider', 'network_type', 'accuracy', 'latitude', 'longitude', 'altitude', 'bearing', 'speed',
                'travelstate', 'blank']
    try:
        with open(f'StudentLife_Dataset/Inputs/sensing/gps/gps_u{number}.csv') as file:
            data = csv.reader(file)
            allData = np.array([lines for lines in data])
        gps = []
        for i in range(1, len(allData)):
            if len(allData[i]) != 0:
                gps.append(allData[i])

        traveled_distance = 0
        traveled_distance_for_day = 0

        for i in range(len(gps)):
            if gps[i][8] != 0 and i != 0 and i != len(gps)-1:
                time = int(gps[i + 1][0]) - int(gps[i - 1][0])
                distance = time * float(gps[i][8])
                traveled_distance += distance
                timestamp = int(gps[i][0])
                local_time = t.strftime("%H:%M:%S", t.localtime(timestamp))
                if (local_time >= "06:00:00") and (local_time <= "18:00:00"):
                    traveled_distance_for_day += distance

        count = 0
        cell_count = 0
        wifi_count = 0
        gps_count = 0

        gps_data = pd.read_csv(f'StudentLife_Dataset/Inputs/sensing/gps/gps_u{number}.csv', header=0, names=col_name)
        for row in gps_data['provider'].index:
            count += 1
            provider = str(gps_data['provider'].get(row))
            if provider == 'gps':
                gps_count += 1
            else:
                if str(gps_data['network_type'].get(row)) == 'cell':
                    cell_count += 1
                elif str(gps_data['network_type'].get(row)) == 'wifi':
                    wifi_count += 1

        cell_type_ratio = cell_count / count
        wifi_type_ratio = wifi_count / count
        gps_type_ratio = gps_count / count

    except FileNotFoundError:
        traveled_distance = None
        cell_type_ratio = None
        wifi_type_ratio = None
        gps_type_ratio = None
        traveled_distance_for_day = None

    features.append(traveled_distance)
    features.append(cell_type_ratio)
    features.append(wifi_type_ratio)
    features.append(gps_type_ratio)
    features.append(traveled_distance_for_day)
    paper_features.append(traveled_distance)
    paper_features.append(traveled_distance_for_day)
    paper_features.append(wifi_type_ratio)


    # We downloaded the full dataset from the link given in the assignment specification.
    # Since the model to compute the sleep duration is not given,
    #   we decided to use the Sleep dataset given under the directory: dataset/dataset/EMA/response/Sleep.
    # According to the file named "EMA_definition.json",
    #   we extracted the "hour", "rate" and "social" data from each file,
    #       where "hour" contains sleep durations, "rate" contains sleep quality ratings
    #           and "social" contains frequencies of feeling sleepy during social activities.
    # After extracting data from the dataset,
    #   we computed the mean sleep duration, mean sleep quality rating and mean tiredness ratings for each student,
    #       i.e.:
    #           37. sleep_duration
    #           38. mean_sleep_rate
    #           39. mean_sleep_tiredness
    try:
        with open(f'StudentLife_Dataset/Inputs/EMA/Sleep/Sleep_u{number}.json') as sleepFile:
            data = json.load(sleepFile)
        sleepData = np.asarray(data)

        hours_list = []
        rates_list = []
        tiredness_list = []

        for sleepDict in sleepData:
            # Check whether the data is invalid
            if len(sleepDict) < 5:
                continue
            hour = int(sleepDict['hour'])
            hours_list.append(hour)
            rate = int(sleepDict['rate'])
            rates_list.append(rate)
            tiredness = int(sleepDict['social'])
            tiredness_list.append(tiredness)

        sleep_duration = sum(hours_list) / len(hours_list)
        mean_sleep_rate = sum(rates_list) / len(rates_list)
        mean_sleep_tiredness = sum(tiredness_list) / len(tiredness_list)

    except FileNotFoundError:
        sleep_duration = None
        mean_sleep_rate = None
        mean_sleep_tiredness = None

    features.append(sleep_duration)
    features.append(mean_sleep_rate)
    features.append(mean_sleep_tiredness)
    paper_features.append(sleep_duration)


    # Compute the ratio of wifi signal levels below the median number of all levels for each student,
    #   since the signal of Internet may have some positive of negative effects on students, i.e.:
    #       40. wifi_ratio
    try:
        wifi_data = pd.read_csv(f'StudentLife_Dataset/Inputs/sensing/wifi/wifi_u{number}.csv', header=0)
        count = 0
        high_level_count = 0
        for level in wifi_data['level']:
            if level is not None:
                count += 1
            if level <= wf_median:
                high_level_count += 1
        wifi_ratio = high_level_count / count

    except FileNotFoundError:
        wifi_ratio = None

    features.append(wifi_ratio)


    # Compute the ratio of high level blue tooth signals for each student,
    #   which is the ratio of the level lower than the median number among the total data, i.e.:
    #       41. bt_ratio
    try:
        bt_data = pd.read_csv(f'StudentLife_Dataset/Inputs/sensing/bluetooth/bt_u{number}.csv', header=0)
        count = 0
        high_level_count = 0
        for level in bt_data['level']:
            if level is not None:
                count += 1
            if level <= bt_median:
                high_level_count += 1
        bt_ratio = high_level_count / count
    except FileNotFoundError:
        bt_ratio = None
    # Append the ratio to the list.
    features.append(bt_ratio)


    # Compute the number of locations each student has been to during the study, i.e.:
    #   42. number_of_locations
    col_name = ['time', 'location', 'blank']
    try:
        location_data = pd.read_csv(f'StudentLife_Dataset/Inputs/sensing/wifi_location/wifi_location_u{number}.csv',
                                    header=0, names=col_name)
        number_of_locations = len(location_data['location'].drop_duplicates())
    except FileNotFoundError:
        number_of_locations = None

    features.append(number_of_locations)


    # Compute mean number of people a student contacted every day, i.e:
    #   43. mean_number_of_ppl_contacted
    try:
        with open(f'StudentLife_Dataset/Inputs/EMA/Social/Social_u{number}.json') as socialFile:
            data = json.load(socialFile)
        socialData = np.asarray(data)

        number_of_ppl_contacted = []

        for socialDict in socialData:
            # Check whether the data is invalid
            if len(socialDict) < 3:
                continue
            else:
                number_of_ppl_contacted.append(int(socialDict['number']))

        try:
            mean_number_of_ppl_contacted = sum(number_of_ppl_contacted) / len(number_of_ppl_contacted)
        except ZeroDivisionError:
            mean_number_of_ppl_contacted = None

    except FileNotFoundError:
        mean_number_of_ppl_contacted = None

    features.append(mean_number_of_ppl_contacted)
    paper_features.append(mean_number_of_ppl_contacted)


    # Append the target values to the list.
    target_values = [flourishing, positive_panas, neagtive_panas]
    for target_data in target_values:
        flag = 0
        for i in range(len(target_data)):
            if target_data[i][0] == f'u{number}':
                features.append(target_data[i][1])
                paper_features.append(target_data[i][1])
                flag += 1
                break
        if flag == 0:
            features.append(None)
            paper_features.append(None)


    # Append the list to the array which stores all features.
    student_features.append(features)
    features_according_to_paper.append(paper_features)



# Create an array to store all features, in the format of:
#                                   student     feature_1   feature_2    feature_3  ... feature_n
#                                      1           x1          x2           x3      ...     xn
#                                      .            .           .            .       .       .
#                                      .            .           .            .       .       .
#                                      .            .           .            .       .       .
#                                      m           x1          x2           x3      ...     xn
# and an array named features_according_to_paper to store all features used in the paper given in the assignment spec.
student_features = []
features_according_to_paper = []

# Compute the median numbers needed.
bluetooth_median = calculate_bluetooth_median()
wifi_median = calculate_wifi_median()

# Pre-process the flourishing scales.
flourishing_data = pre_process_flourishing_scale()

# Pre-process the positive and negative PANAS socres.
positive_panas_data, negative_panas_data = pre_process_panas()

# Use a for loop to read the input data of all students,
# and call the pre_process function to deal with the raw data.
for n in range(60):
    pre_process(n, bluetooth_median, wifi_median, flourishing_data, positive_panas_data, negative_panas_data)


# Save the pre-processed data as a csv file.
all_features_result = []
for line in student_features:
    none_flag = 0
    for item in line:
        if item is None:
            none_flag += 1
            break
    if not none_flag:
        all_features_result.append(line)
array = np.asarray(all_features_result)
df = pd.DataFrame(all_features_result)
df.to_csv("all_features.csv", index=False)

paper_features_result = []
for line in features_according_to_paper:
    none_flag = 0
    for item in line:
        if item is None:
            none_flag += 1
            break
    if not none_flag:
        paper_features_result.append(line)
array = np.asarray(paper_features_result)
df = pd.DataFrame(paper_features_result)
df.to_csv("paper_features.csv", index=False)

print("FINISHED")


