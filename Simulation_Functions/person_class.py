import sys
import simpy
from Simulation_Functions.DES_functions import *
################################################################# MODEL #####################################################################################
class User(object):
    def __init__(self, env,i, initial_done, Person_Dict, gen_dict, params):        
        #creates a new user as process that
        self.env = env
        self.action = env.process(self.setUser(Person_Dict,env,gen_dict,params))
        self.user_type = 'nru' #User types: nru -non-relapsed users, rnu - recovering non-user, ru - relapsed users, d - deceased
        self.isAlive = True #status type: alive = True, dead=False, used to exit user loop
        self.num = i #numbered person
        self.timeofNonOpioidDeath, self.EnterAge = alldeath_time(initial_done, gen_dict["alldeath_gen"], params) 
        self.timeofNonOpioidDeath = self.timeofNonOpioidDeath + self.env.now
        self.timeofFatalOD = OR_death_time(self.env.now, params,gen_dict["death_gen"]) + self.env.now
        # print('New user %d Created. %s at time %d.' % (self.num, self.getUserType(), self.env.now))
    
    def setUser(self, Person_Dict,env,gen_dict,params):
        ########### Parameters for Run ###############
        n_runs = params["n_runs"]
        num_years = params["num_years"]
        days_per_year = params["days_per_year"]
        hospital_encounter_thres = params["hospital_encounter_thres"]
        hospital_encounter_thres_base = params["HE_thres_baseline"]
        ED2R_start_time = params["ED2R_start_time"] 
        MARI_threshold = params["MARI_thres"]
        MARI_start_time = params["MARI_start_time"]
        CM_threshold = params["CM_thres"]
        CM_start_time = params["CM_start_time"]
        MARI_gen = gen_dict["MARI_gen"]
        hosp_sub_gen = gen_dict["hosp_sub_gen"]
        CM_gen = gen_dict["CM_gen"]
        
        ####### person Stuff ##############
        Person_Dict[self.num]["isAlive"] = True
        Person_Dict[self.num]["EnterAge"] = self.EnterAge
        Person_Dict[self.num]["Num_inTreatment"] = 0
        Person_Dict[self.num]["List_Times_inTreatment"] = []
        Person_Dict[self.num]["List_Treat_ServiceTimes"] = []
        Person_Dict[self.num]["List_Treat_ExitTimes"] = []
        Person_Dict[self.num]["List_InactiveTreat_ServiceTimes"] = []
        Person_Dict[self.num]["Num_Crimes"] = 0 
        Person_Dict[self.num]["List_Times_ofCrime"] = []
        Person_Dict[self.num]["List_Crime_ServiceTimes"] = []
        Person_Dict[self.num]["List_Crime_ExitTimes"] = []
        try:
            if Person_Dict[self.num]["List_Crime_Type"]:
                pass
        except: 
            Person_Dict[self.num]["List_Crime_Type"] = []
        Person_Dict[self.num]["List_InactiveCrime_ServiceTimes"] = []
        Person_Dict[self.num]["Num_Hosp"] = 0
        Person_Dict[self.num]["List_Times_inHosp"] = []
        Person_Dict[self.num]["List_Hosp_ServiceTimes"] = []
        Person_Dict[self.num]["List_Hosp_ExitTimes"] = []
        Person_Dict[self.num]["List_Hosp_ExitNext"] = []
        Person_Dict[self.num]["List_InactiveHosp_ServiceTimes"] = []
        Person_Dict[self.num]["Num_Inactive_only"] = 0
        Person_Dict[self.num]["List_Times_Inactive_only"] = []
        Person_Dict[self.num]["List_InactiveOnly_ServiceTimes"] = []
        Person_Dict[self.num]["List_Relapse_Time"] = []
        Person_Dict[self.num]["OpioidDeathT"] = None
        Person_Dict[self.num]["NonOpioidDeathT"] = None
        Person_Dict[self.num]["OpioidDeath"] = None
        Person_Dict[self.num]["List_Crime_ExitNext"] = []
        if "Next_Interrupt" in Person_Dict[self.num]:
            Person_Dict[self.num]["PrevState"] = Person_Dict[self.num]["Next_Interrupt"]
            Person_Dict[self.num]["Time_FirstActive"] = float('nan')
        else:
            Person_Dict[self.num]["Next_Interrupt"] = ''
            Person_Dict[self.num]["PrevState"] = "active"
        while self.isAlive: #continues until individual exits the simlation via overdose death.           
            # Initial Active Use
            if self.user_type == 'nru':
                Person_Dict[self.num]["TimeofFatalD"] = self.timeofNonOpioidDeath 
                Person_Dict[self.num]["TimeofFatalOD"] = self.timeofFatalOD 
                # print('person %d has become a %s and activated thier disease at time %f and has a scheduled death time at time %f' % (self.num, self.getUserType(), self.env.now,self.timeofFatalOD + env.now))
            # Relapsed Active Use
            else:
                self.user_type = 'ru'
                self.timeofFatalOD = OR_death_time(self.env.now,params, gen_dict["death_gen"]) + self.env.now # see fxn def #
                Person_Dict[self.num]["TimeofFatalOD"] = self.timeofFatalOD 
                Person_Dict[self.num]["PrevState"] = "active"
                # print('person %d has become a %s and activated thier disease at time %f and has a scheduled death time at time %f' % (self.num, self.getUserType(), self.env.now,self.timeofFatalOD + env.now))
            # Process to set Deceased
            try:
                yield self.env.process(self.NextEvent(env, Person_Dict[self.num], gen_dict, params))
                yield self.env.timeout(0.00001) 
                if self.user_type == 'd':
                    sys.exit("Error: This person is already deceased")
                else:
                    if self.env.now > float(days_per_year*num_years) or min(self.timeofFatalOD, self.timeofNonOpioidDeath) > float(days_per_year*num_years):
                        # print(self.env.now, ">?", float(days_per_year*num_years))
                        # print(self.timeofFatalOD, ">?", float(days_per_year*num_years))
                        # print(self.timeofNonOpioidDeath, ">?",float(days_per_year*num_years))
                        # print("")
                        pass
                    else:
                        self.user_type = 'd'
                        # print('person %d is %s and decesed time %f.' % (self.num,self.getUserType(), self.env.now))
                        if self.timeofFatalOD < self.timeofNonOpioidDeath:
                            Person_Dict[self.num]["OpioidDeath"] = True
                            Person_Dict[self.num]["OpioidDeathT"] = self.env.now
                        else:
                            Person_Dict[self.num]["OpioidDeath"] = False
                            Person_Dict[self.num]["NonOpioidDeathT"] = self.env.now
                        Person_Dict[self.num]["isAlive"] = False
                        Person_Dict[self.num]["State"] = "death"
                        self.isAlive= False
            # Interuptions from setting Deceased
            except simpy.Interrupt as interrupt:
                if self.user_type == 'nru' or self.user_type == 'ru':
                    if self.user_type == 'nru':
                        #Amount of time person is originally active until first state
                        if "Time_FirstActive" in Person_Dict[self.num]:
                            pass
                        else:
                            Person_Dict[self.num]["Time_FirstActive"] = self.env.now - Person_Dict[self.num]["arrivalT"]
                    self.user_type = 'rnu' #set user type to a recovering non-user. Person goes to treatment either via treatment or jail
                    enter_time = self.env.now
                    # Treatment Interruption
                    if interrupt.cause == 'treatment':
                        s_time = service_time(interrupt.cause, gen_dict["service_gen"],params) # Service time depending on interruption type, see fxn def 
                        r_time = relapse_time(interrupt.cause, Person_Dict[self.num]["arrivalT"], env.now, gen_dict, params) # see fxn def 
                        self.sobriety_duration = s_time + r_time
                        self.timeofFatalOD = 10000000 #cannot die from opioid related cause while in treatment
                        Person_Dict[self.num]["State"] = "Treatment"
                        Person_Dict[self.num]["TimeofFatalOD"] = self.timeofFatalOD + self.env.now
                        Person_Dict[self.num]["Num_inTreatment"] = Person_Dict[self.num]["Num_inTreatment"] + 1
                        Person_Dict[self.num]["List_Times_inTreatment"].append(enter_time)
                        Person_Dict[self.num]["List_Treat_ServiceTimes"].append(s_time)
                        Person_Dict[self.num]["List_Treat_ExitTimes"].append(enter_time+s_time)
                        Person_Dict[self.num]["List_InactiveTreat_ServiceTimes"].append(r_time)
                        Person_Dict[self.num]["List_Relapse_Time"].append(enter_time + s_time + r_time)
                        # print('person %d is a %s at time %f and is in treatment. They will be in treatment for %f time and stop treatment at time %f. After treatment, their opioid use will remain inactive for %f time and will reactivate at time %f' % (self.num,self.getUserType(), enter_time,s_time,(enter_time+s_time), r_time, (enter_time+s_time+r_time)))
                    # Crime Interruption
                    elif interrupt.cause == 'crime':
                        self.timeofFatalOD = 10000000 #cannot die from opioid related cause until relapse
                        Person_Dict[self.num]["State"] = "crime"
                        Person_Dict[self.num]["TimeofFatalOD"] = self.timeofFatalOD + self.env.now
                        Person_Dict[self.num]["Num_Crimes"] = Person_Dict[self.num]["Num_Crimes"] + 1
                        Person_Dict[self.num]["List_Times_ofCrime"].append(enter_time)
                        rand_number = MARI_gen.random()
                        if rand_number <= MARI_threshold and env.now > MARI_start_time and Person_Dict[self.num]["List_Crime_Type"][-1] =="Ocrime":
                            Person_Dict[self.num]["List_Crime_ExitNext"].append("treatMARI")
                            Person_Dict[self.num]["List_Crime_ServiceTimes"].append(0.001)
                            Person_Dict[self.num]["List_Crime_ExitTimes"].append(enter_time+ 0.001)
                            Person_Dict[self.num]["nextTreatT"] = enter_time + 0.001
                            #print('person %d has had an Arrest at time %f, and will go stright to treatment after a service time of %f at time %f.' % (self.num, enter_time, 0.001, Person_Dict[self.num]["nextTreatT"]))
                            #recaluclate service and relapse times as the individual is now in treatment
                            s_time = service_time('treatment',gen_dict["service_gen"], params) # Service time depending on interruption type, see fxn def 
                            r_time = relapse_time('treatment', Person_Dict[self.num]["arrivalT"], env.now,gen_dict, params) # see fxn def
                            #print('person %d is a %s at time %f and is in treatment. They will be in treatment for %f time and stop treatment at time %f. After treatment, their opioid use will remain inactive for %f time and will reactivate at time %f' % (self.num,self.getUserType(), enter_time,s_time,(enter_time+s_time), r_time, (enter_time+s_time+r_time)))
                            Person_Dict[self.num]["State"] = "Treatment"
                            Person_Dict[self.num]["Num_inTreatment"] = Person_Dict[self.num]["Num_inTreatment"] + 1
                            Person_Dict[self.num]["List_Times_inTreatment"].append(enter_time + 0.001)
                            Person_Dict[self.num]["List_Treat_ServiceTimes"].append(s_time)
                            Person_Dict[self.num]["List_Treat_ExitTimes"].append(enter_time+s_time)
                            Person_Dict[self.num]["List_InactiveTreat_ServiceTimes"].append(r_time)
                            self.sobriety_duration = s_time + r_time
                            Person_Dict[self.num]["List_Relapse_Time"].append(enter_time + s_time + r_time)
                        else:
                            rand_num2 = CM_gen.random()
                            s_time = service_time(interrupt.cause,gen_dict["service_gen"], params) # Service time depending on interruption type, see fxn def 
                            if rand_num2 <= CM_threshold and (env.now + s_time) > CM_start_time:
                                Person_Dict[self.num]["List_Crime_ExitNext"].append("treatCM")
                                Person_Dict[self.num]["nextTreatT"] = enter_time + s_time + 0.001
                                Person_Dict[self.num]["List_Crime_ServiceTimes"].append(s_time)
                                Person_Dict[self.num]["List_Crime_ExitTimes"].append(enter_time+s_time)
                                Person_Dict[self.num]["Next_Interrupt"] = 'treat' 
                                self.sobriety_duration = s_time 
                            else:      
                                Person_Dict[self.num]["List_Crime_ExitNext"].append("inactive")
                                r_time = relapse_time(interrupt.cause, Person_Dict[self.num]["arrivalT"], env.now,gen_dict, params) # see fxn def 
                                self.sobriety_duration = s_time + r_time
                                Person_Dict[self.num]["List_Crime_ServiceTimes"].append(s_time)
                                Person_Dict[self.num]["List_Crime_ExitTimes"].append(enter_time+s_time)
                                Person_Dict[self.num]["List_InactiveCrime_ServiceTimes"].append(r_time)
                                Person_Dict[self.num]["List_Relapse_Time"].append(enter_time + s_time + r_time)
                            # print('person %d is a %s at time %f and has been arrested. They will serve %f time and be relased at time %f. After CJ release, their opioid use will remain inactive for %f time and will reactivate at time %f.' % (self.num,self.getUserType(), enter_time,s_time,(enter_time+s_time), r_time, (enter_time+s_time+r_time)))
                    # Hospital Encounter Interruption
                    elif interrupt.cause == 'hosp':
                        self.timeofFatalOD = 10000000 #cannot die from opioid related cause until relapse
                        s_time = service_time(interrupt.cause,gen_dict["service_gen"], params) # Service time depending on interruption type, see fxn def 
                        Person_Dict[self.num]["State"] = "hosp"
                        Person_Dict[self.num]["Num_Hosp"] = Person_Dict[self.num]["Num_Hosp"] + 1
                        Person_Dict[self.num]["List_Times_inHosp"].append(enter_time)
                        Person_Dict[self.num]["List_Hosp_ServiceTimes"].append(s_time) 
                        Person_Dict[self.num]["List_Hosp_ExitTimes"].append(enter_time+s_time)
                        rand_number = hosp_sub_gen.random()
                        if env.now > ED2R_start_time:
                            HE_thres = hospital_encounter_thres
                        else:
                            HE_thres = hospital_encounter_thres_base
                        # print('person %d has had a Hospital Encounter at time %f, and had a random number of %f.' % (self.num, enter_time, rand_number))
                        if rand_number < HE_thres[0]: # probability of getting arrested after hosp encounter
                            Person_Dict[self.num]["nextCrimeT"] = enter_time+ s_time + 0.001
                            Person_Dict[self.num]["List_Hosp_ExitNext"].append("crime")
                            Person_Dict[self.num]["List_Crime_Type"].append("Ocrime")
                            Person_Dict[self.num]["Next_Interrupt"] = 'crime'
                            self.sobriety_duration = s_time
                            # print('person %d has had a Hospital Encounter at time %f, and will go stright to arrest after a service time of %f at time %f.' % (self.num, enter_time, s_time, Person_Dict[self.num]["nextCrimeT"])) 
                        elif rand_number < HE_thres[1]: # probability of starting treatment after hosp encounter
                            Person_Dict[self.num]["nextTreatT"] = enter_time + s_time + 0.001
                            Person_Dict[self.num]["Next_Interrupt"] = 'treat' 
                            Person_Dict[self.num]["List_Hosp_ExitNext"].append("treat")
                            self.sobriety_duration = s_time 
                            # print('person %d has had a Hospital Encounter at time %f, and will go stright to treatment after a service time of %f at time %f.' % (self.num, enter_time, s_time, Person_Dict[self.num]["nextTreatT"])) 
                        elif rand_number < HE_thres[2]:  # probability of dying of fatal overdose after hosp encounter
                            Person_Dict[self.num]["TimeofFatalOD"] = enter_time + s_time + 0.001
                            Person_Dict[self.num]["PrevState"] = "hosp"
                            Person_Dict[self.num]["List_Hosp_ExitNext"].append("fatal")
                            self.sobriety_duration = s_time 
                            # print('person %d has had a Hospital Encounter at time %f, and will die after a service time of %f at time %f.' % (self.num, enter_time, s_time, Person_Dict[self.num]["TimeofFatalOD"])) 
                        else:
                            Person_Dict[self.num]["List_Hosp_ExitNext"].append("inactive")
                            r_time = relapse_time(interrupt.cause, Person_Dict[self.num]["arrivalT"], env.now,gen_dict, params) # see fxn def 
                            self.sobriety_duration = s_time + r_time
                            Person_Dict[self.num]["List_Relapse_Time"].append(enter_time + s_time + r_time)
                            Person_Dict[self.num]["List_InactiveHosp_ServiceTimes"].append(r_time)
                            # print('person %d has had a Hospital Encounter at time %f. They have a service time of %f and will be released at time %f. After Hospital release, their opioid use will remain inactive for %f time and will reactivate at time %f.' % (self.num, enter_time, s_time,(enter_time+s_time), r_time, (enter_time+s_time+r_time)))
                    # Inactive Interruption
                    elif interrupt.cause == 'inactive':
                        s_time = service_time(interrupt.cause, gen_dict["service_gen"], params) # Service time depending on interruption type, see fxn def 
                        r_time = relapse_time(interrupt.cause, Person_Dict[self.num]["arrivalT"], env.now, gen_dict, params) # see fxn def 
                        self.sobriety_duration = r_time
                        self.timeofFatalOD = 10000000 #cannot die temporarily until relapse
                        Person_Dict[self.num]["State"] = "inactive"
                        Person_Dict[self.num]["TimeofFatalOD"] = self.timeofFatalOD + self.env.now
                        Person_Dict[self.num]["Num_Inactive_only"] = Person_Dict[self.num]["Num_Inactive_only"] + 1
                        Person_Dict[self.num]["List_Times_Inactive_only"].append(enter_time)
                        Person_Dict[self.num]["List_Relapse_Time"].append(enter_time + r_time)
                        Person_Dict[self.num]["List_InactiveOnly_ServiceTimes"].append(r_time)
                        # print('person %d is a %s at time %f and stoped opioid use. Their opioid use will remain inactive for %f time, and will reactivate at time %f.' % (self.num,self.getUserType(), enter_time,r_time,(enter_time+r_time)))
                    # Other Interruption
                    else:
                        print("unknown interuption cause: ",interrupt.cause)
                    
                    if self.timeofNonOpioidDeath < (self.env.now + self.sobriety_duration) and self.timeofNonOpioidDeath < days_per_year*num_years :
                        timeleft = self.timeofNonOpioidDeath - self.env.now
                        if self.timeofNonOpioidDeath < (self.env.now + s_time): #inactive has a service time of 0
                            if interrupt.cause == "crime":
                                if Person_Dict[self.num]["List_Crime_ExitNext"][-1] == "treatMARI": #should be if
                                    Person_Dict[self.num]["PrevState"] = "treat"
                                    del Person_Dict[self.num]["List_Relapse_Time"][-1]
                                    del Person_Dict[self.num]["List_Treat_ExitTimes"][-1] 
                                    Person_Dict[self.num]["List_Treat_ServiceTimes"][-1] =  self.timeofNonOpioidDeath - Person_Dict[self.num]["List_Times_inTreatment"][-1]
                                elif Person_Dict[self.num]["List_Crime_ExitNext"][-1] == "treatCM":
                                    Person_Dict[self.num]["PrevState"] = Person_Dict[self.num]["List_Crime_Type"][-1]
                                    del Person_Dict[self.num]["List_Crime_ExitTimes"][-1]
                                    Person_Dict[self.num]["List_Crime_ServiceTimes"][-1] =  self.timeofNonOpioidDeath - Person_Dict[self.num]["List_Times_ofCrime"][-1]
                                else:    
                                    Person_Dict[self.num]["PrevState"] = Person_Dict[self.num]["List_Crime_Type"][-1]
                                    del Person_Dict[self.num]["List_Relapse_Time"][-1]
                                    del Person_Dict[self.num]["List_Crime_ExitTimes"][-1]
                                    del Person_Dict[self.num]['List_InactiveCrime_ServiceTimes'][-1]
                                    Person_Dict[self.num]["List_Crime_ServiceTimes"][-1] =  self.timeofNonOpioidDeath - Person_Dict[self.num]["List_Times_ofCrime"][-1]
                            elif interrupt.cause == "treatment":
                                Person_Dict[self.num]["PrevState"] = "treat"
                                del Person_Dict[self.num]["List_Relapse_Time"][-1]
                                del Person_Dict[self.num]["List_Treat_ExitTimes"][-1] 
                                del Person_Dict[self.num]['List_InactiveTreat_ServiceTimes'][-1]
                                Person_Dict[self.num]["List_Treat_ServiceTimes"][-1] =  self.timeofNonOpioidDeath - Person_Dict[self.num]["List_Times_inTreatment"][-1]
                            elif interrupt.cause == "hosp":
                                Person_Dict[self.num]["PrevState"] = "hosp"
                                del Person_Dict[self.num]["List_Hosp_ExitTimes"][-1]
                                Person_Dict[self.num]["List_Hosp_ServiceTimes"][-1] =  self.timeofNonOpioidDeath - Person_Dict[self.num]["List_Times_inHosp"][-1]
                                if Person_Dict[self.num]["List_Hosp_ExitNext"][-1] == "inactive":
                                    del Person_Dict[self.num]["List_Relapse_Time"][-1]
                                    del Person_Dict[self.num]['List_InactiveHosp_ServiceTimes'][-1]
                                # elif Person_Dict[self.num]["List_Hosp_ExitNext"][-1] == "fatal": if someone has a comorbiity and dies from fatal even through they wer ein teh hpsial for opioid reason cause of death is non-opioid.
                            else:
                                print("OTHER?!?!?")
                        else:
                            Person_Dict[self.num]["PrevState"] = "inactive"
                            del Person_Dict[self.num]["List_Relapse_Time"][-1]
                        timeleft = max(timeleft,0)
                        yield self.env.timeout(timeleft)
                        Person_Dict[self.num]["OpioidDeath"] = False
                        Person_Dict[self.num]["NonOpioidDeathT"] = self.env.now
                        Person_Dict[self.num]["isAlive"] = False
                        Person_Dict[self.num]["State"] = "death"
                        self.isAlive=False
                        # print('person %d has died at time %f.' % (self.num, self.env.now))
                    else:
                        # yield self.env.timeout(self.sobriety_duration)
                        yield self.env.timeout(self.sobriety_duration) #interupts process to go to death during sobriety.                    
                        Person_Dict[self.num]["State"] = "active"
                        if interrupt.cause == 'hosp':
                            if Person_Dict[self.num]["List_Hosp_ExitNext"][-1] == "fatal":
                                Person_Dict[self.num]["OpioidDeath"] = True
                                Person_Dict[self.num]["isAlive"] = False
                                # print('person %d is %s at time %f.' % (self.num,self.getUserType(), self.env.now))
                                Person_Dict[self.num]["OpioidDeathT"] = self.env.now
                                Person_Dict[self.num]["State"] = "death"
                                self.isAlive= False
                        if Person_Dict[self.num]["Next_Interrupt"] == '':
                            self.user_type = 'ru'
                            # print('person %d has become a %s and activated their disease at time %f.' % (self.num, self.getUserType(), self.env.now))
                else:
                    sys.exit("Error: This person is already a recovering non-user")
    def NextEvent(self, env, Person_Dict, gen_dict,params):
        if Person_Dict["Next_Interrupt"] == 'treat':
            Person_Dict["Next_Interrupt"] = ''
            self.action.interrupt('treatment')
            yield self.env.timeout(0.00001)   #allows sobriety_duration time to be calculated
            yield env.timeout(self.sobriety_duration) #interupts next event process 
        elif Person_Dict["Next_Interrupt"] == 'crime':
            Person_Dict["Next_Interrupt"] = ''
            self.action.interrupt('crime')
            yield self.env.timeout(0.00001)   #allows sobriety_duration time to be calculated
            yield env.timeout(self.sobriety_duration) #interupts next event process 
        elif Person_Dict["Next_Interrupt"] == 'hosp':
            Person_Dict["Next_Interrupt"] = ''
            self.action.interrupt('hosp')
            yield self.env.timeout(0.00001)   #allows sobriety_duration time to be calculated
            yield env.timeout(self.sobriety_duration) #interupts next event process 
        elif Person_Dict["Next_Interrupt"] == 'inactive':
            Person_Dict["Next_Interrupt"] = ''
            self.action.interrupt('inactive')       
            yield self.env.timeout(0.00001)   #allows sobriety_duration time to be calculated
            yield env.timeout(self.sobriety_duration) #interupts next event process  
        else:
            #Calculate all next possible event times
            OCtime = Ocrime_time(gen_dict["Ocrime_gen"], params) #see fx def #
            try:
                nonOCtime = Person_Dict["nextnonOCrimeT"]
                if Person_Dict["nextCrimeT"] < env.now:
                    nonOCtime = nonOcrime_time(gen_dict["nonOcrime_gen"], params) #see fx def #
            except:
                nonOCtime = nonOcrime_time(gen_dict["nonOcrime_gen"], params) #see fx def #
            Person_Dict["nextnonOCrimeT"] = nonOCtime + env.now
            Person_Dict["nextCrimeT"] = min(OCtime,nonOCtime) + env.now
            # print('person %d has a scheduled next Crime at time %f' % (i, Ctime+env.now))
            Ttime = treat_time(gen_dict["treat_gen"], params) #see fxn def #
            Person_Dict["nextTreatT"] = Ttime + env.now
            # print('person %d has a scheduled next treatment at time %f' % (i, Ttime+env.now))
            Htime = hosp_time(gen_dict["hosp_gen"], params) #see fxn def #
            Person_Dict["nextHospT"] = Htime + env.now
            # print('person %d has a scheduled next Hospital Encounter at time %f' % (i, Htime+env.now))
            Itime = inactive_time(gen_dict["inactive_gen"], params) 
            Person_Dict["NextInactiveT"] = Itime + env.now
            # print('person %d has a scheduled next Inactive Stage at time %f' % (i, Itime+env.now))
            Dtime = self.timeofNonOpioidDeath - env.now
            ODtime = self.timeofFatalOD - env.now
            #select the minimum time to be the next event
            time = min(nonOCtime, OCtime,Ttime,Htime,Itime, ODtime,Dtime)            
            # get index of smallest item in list
            list_times = (nonOCtime, OCtime,Ttime,Htime,Itime,ODtime,Dtime)
            X = list_times.index(min(list_times))
            Person_Dict["NextEventType"] = X
            Person_Dict["TimeUnitNextEvent"] = time
            # print('person %d has a scheduled next Event at time %f of Type %f' % (i, time+env.now, X))
            time = max(time,0)
            yield self.env.timeout(time)
            # if self.self_type == 'nru' or self.self_type=='ru' and self.isAlive:
            if X == 0:
                Person_Dict["List_Crime_Type"].append("nonOcrime")
                self.action.interrupt('crime')        
                # print('person %d is now being arrested at time %f of Type %f' % (i, env.now, X))            
            elif X== 1:
                Person_Dict["List_Crime_Type"].append("Ocrime")
                self.action.interrupt('crime')        
                # print('person %d is now being arrested at time %f of Type %f' % (i, env.now, X))            
            elif X == 2:
                self.action.interrupt('treatment')
                # print('person %d is now starting treatment at time %f of Type %f' % (i, env.now, X))
            elif X == 3:
                self.action.interrupt('hosp')
                # print('person %d is now going to hosp at time %f of Type %f' % (i, env.now, X))
            elif X == 4:
                self.action.interrupt('inactive')
                # print('person %d in now inactive at time %f of Type %f' % (i, env.now, X))
######################### User Arrivals ######################################################
def user_arrivals(env, i,initial_done,gen_dict, params, Persons):
    Person_Dict = Persons["dict"] 
    Person = Persons["indv"]
    arrival_gen = gen_dict["arrival_gen"]
    lam_user_arrival = params["lam_user_arrival"]
    #Create new users until the sim end time is reached
    while True:
        time = arrival_time(arrival_gen, lam_user_arrival) #see fxn def # 
        #print("time until next arrival", expo)
        Person_Dict[i] = {}
        Person_Dict[i]["arrivalT"] = time + env.now
        time = max(time,0)
        yield env.timeout(time)
        #can add conditions to delay person arrival or assign them to services / groups see  #https://simpy.readthedocs.io/en/latest/examples/movie_renege.html?highlight=arrivals#movie-renege,
        Person[i] = User(env,i, initial_done, Person_Dict, gen_dict,params)
        i = i + 1
