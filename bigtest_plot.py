import tensorflow as tf
import numpy as np
import random_graph_gen as rd
import network_topology_gen as nt
# import memory_pool as mp
# import gcn_layer_with_pooling_v2 as gcn
import environment_gen as en
import time
import random
import agent_with_batch_input as ag
import multiprocessing as mp
import os
import sys
import copy
import logging
import pickle
from scipy.stats import poisson
from decimal import Decimal
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')


os.environ['CUDA_VISIBLE_DEVICES']=''


ENTROPY_WEIGHT = 0.5
EPS = 1e-6
INPUT_FEATURES = 3
EXTRACTED_FEATURES = INPUT_FEATURES *20
SNODE_SIZE = 100
VNODE_FEATURES_SIZE = 3
#NUM_AGENT = mp.cpu_count()#4 or 8 i guess
NUM_AGENT=0
ORDERS = 3
GAMMA=0.99
ALIVE_TIME=50000


NEW_DIR="/home/yanzx1993/new_not_averaged/"
#G,node_attr,wmin,wmax= rd.make_weighted_random_graph(SNODE_SIZE, 0.5, "normal",1)
#pickle.dump(G,open(NEW_DIR+"G.var",'wb'))
#pickle.dump(node_attr,open(NEW_DIR+"node_attr.var",'wb'))
#pickle.dump(wmin,open(NEW_DIR+"wmin.var",'wb'))
#pickle.dump(wmax,open(NEW_DIR+"wmax.var",'wb'))
#G=pickle.load(open('G.var','rb'))
GOOD_DIR="/home/yanzx1993/good_models/"

G=pickle.load(open(NEW_DIR+"G.var",'rb'))
print("total edges:",G.number_of_edges())
node_attr=pickle.load(open(NEW_DIR+"node_attr.var",'rb'))
wmin=pickle.load(open(NEW_DIR+"wmin.var",'rb'))
wmax=pickle.load(open(NEW_DIR+"wmax.var",'rb'))
#list1=None
#list2=None
#list3=None
#list4=None
#list5=None
#list1=pickle.load(open(NEW_DIR+"rate_list.var",'rb'))
list1=pickle.load(open(NEW_DIR+"rate_list_sparse.var",'rb'))
list2=pickle.load(open(NEW_DIR+"edge_list.var",'rb'))
list3=pickle.load(open(NEW_DIR+"node_list.var",'rb'))
list4=pickle.load(open(NEW_DIR+"all_list.var",'rb'))
list5=pickle.load(open(NEW_DIR+"size_list.var",'rb'))

ENVIRONMENT_LIST=list()
LAPLACIAN = rd.make_laplacian_matrix(G)
LAPLACIAN_LIST = rd.make_laplacian_list(G, SNODE_SIZE, ORDERS)
LAPLACIAN_TENSOR = np.stack(LAPLACIAN_LIST)
RANDOM_SEED=93
MAX_ITER=1000

N_WORKERS=mp.cpu_count()
epoch=0
#NN_MODEL=None
NN_MODEL=NEW_DIR+'_ep_59000.ckpt'
LOG_DIR = "/home/yanzx1993/VNR_A3C_SUMMARY"
MODEL_DIR="/home/yanzx1993/VNR_A3C_MODEL"
LOG_FILE="/home/yanzx1993/VNR_A3C_LOG/LOG"



def random_gen(rate):
    rv=poisson(rate)
    plist=[]
    for i in range(int(rate*100+1)):
        plist.append(rv.cdf(i))
    rnd=np.random.rand()
    for i in range(len(plist)):
        if(rnd<=plist[i]):
            #print(i)
            return i

def rate_list_gen():
    if(list1!=None):
        return list1
    rate_list=list()
    generated=2000
    rate=0.04
    while(rate<=0.16):
        c_graph=copy.deepcopy(G)
        env=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",rate=rate)
        testing_list=["rate_"+str.format("{:.2f}",rate),env.generate_VNR_list(generated)]
        rate_list.append(testing_list)
        print("generated "+"rate_"+str.format("{:.2f}",(rate))+" success")
        rate+=0.02
        generated+=1000
    pickle.dump(rate_list,open(NEW_DIR+"rate_list_sparse.var",'wb'))
    return rate_list




def res_list_gen():
    if(list2!=None):
        return list2,list3,list4
    edge_list=list()
    node_list=list()
    all_list=list()
    generated=2000
    edge_up=20
    node_up=20
    edge_low=30
    node_low=30
    while(edge_low<=100):
        c_graph=copy.deepcopy(G)
        default="default_"+str(edge_up)+"_"+str(edge_low)+"_"+str(node_up)+"_"+str(node_low)
        env=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",VNR_type=default)
        name="edge_"+str(edge_up)+"_"+str(edge_low)
        testing_list=[name,env.generate_VNR_list(generated)]
        edge_list.append(testing_list)
        edge_low+=10
        print("generated "+name+" success")
    pickle.dump(edge_list,open(NEW_DIR+"edge_list.var",'wb'))
    edge_low=30
    while(node_low<=100):
        c_graph=copy.deepcopy(G)
        default="default_"+str(edge_up)+"_"+str(edge_low)+"_"+str(node_up)+"_"+str(node_low)
        env=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",VNR_type=default)
        name="node_"+str(node_up)+"_"+str(node_low)
        testing_list=[name,env.generate_VNR_list(generated)]
        node_list.append(testing_list)
        node_low+=10
        print("generated "+name+" success")
    pickle.dump(node_list,open(NEW_DIR+"node_list.var",'wb'))
    node_low=30

    while(node_low<=100):
        c_graph=copy.deepcopy(G)
        default="default_"+str(edge_up)+"_"+str(edge_low)+"_"+str(node_up)+"_"+str(node_low)
        env=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",VNR_type=default)
        name="all_"+str(node_up)+"_"+str(node_low)
        testing_list=[name,env.generate_VNR_list(generated)]
        all_list.append(testing_list)
        node_low+=10
        edge_low+=10
        print("generated "+name+" success")
    pickle.dump(all_list,open(NEW_DIR+"all_list.var",'wb'))
    node_low=30
    return edge_list,node_list,all_list


def size_list_gen():
    if(list5!=None):
        return list5
    size_list=list()
    generated=2000
    up=2
    low=10
    while(low<=20):
        c_graph=copy.deepcopy(G)
        size="size_"+str(up)+"_"+str(low)
        env=en.Environment(size, SNODE_SIZE,1000000,imported_graph=[c_graph,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing")
        testing_list=[size,env.generate_VNR_list(generated)]
        size_list.append(testing_list)
        low+=2
        print("generated "+size+" success")
    pickle.dump(size_list,open(NEW_DIR+"size_list.var",'wb'))
    return size_list


def test_env_list(test_name,actor,sess,test_list,embedding="hybrid",type="static",epoch=0):
    x_list=list()
    y0_list=list()
    y1_list=list()
    y2_list=list()
    edge_list=list()
    node_list=list()
    name_list=list()
    acc_list=list()
    suc_list=list()
    for j in range(len(test_list)):
        master_graph=copy.deepcopy(G)
        name=test_list[j][0]
        sublist=test_list[j][1]
        generated=len(sublist)
        env=en.Environment(name, SNODE_SIZE,1000000,imported_graph=[master_graph,node_attr,wmin,wmax],link_embedding_type=embedding,phase="Testing")
        env.testing_VNR_list=copy.deepcopy(sublist)
        snode_sum=env.substrate_network.snode_sum
        sedge_sum=env.substrate_network.edge_weights_sum
        x=list()
        #x.append(0)
        y0=list()
        y1=list()
        y2=list()
        edge=list()
        node=list()
        success=0
        i=0

        while i < generated:
            # test_time=env.time
            # env.release_resource(test_time)
            s, v = env.get_state()
            # print("state acquired.")
            # snode_batch.append(s)
            # vnode_batch.append(v)
            env.snode_state = s
            env.vnode_state = v
            action_prob = actor.predict(s, v)
            # print("current action prob:",action_prob)
            # print(str.format("worker {0} current action prob:{1}",worker_index,action_prob))
            if(type=="static"):
                action_one_hot, action_pick = actor.pick_action(action_prob, env.substrate_network.attribute_list[2]["attributes"],
                                                            sess, phase="Testing")
            else:
                action_one_hot, action_pick = actor.random_pick_action()
            # print("action_pick:",action_pick)
            is_terminal, failure, reward = env.perform_action(action_pick)
            now = time.clock()

            if is_terminal == 1:
                success += 0
                # print(str.format("worker {0} failed.",worker_index))
            elif is_terminal == 2:
                success += 1
                # print(str.format("worker {0} success.",worker_index))
            # if(events==0):
            # step+=1
            # env.time=step
            if is_terminal != 0:
                i += 1
                env.VNR_counter += 1
                env.release_resource(env.time)
            if is_terminal != 0 and env.time>=2000:
                x.append(int(env.time))
                vnode_sum=0
                vedge_sum=0
                for k in range(len(env.assigned_VNR_list)):
                    ve=env.assigned_VNR_list[k].assigned_link
                    vn=env.assigned_VNR_list[k].assigned_node
                    for m in range(len(vn)):
                        vnode_sum+=vn[m][1]
                    for n in range(len(ve)):
                        #print("ve:",ve[n][2])
                        #print("max:",env.substrate_network.max_bandwidth)
                        vedge_sum+=ve[n][2]
                #print("len:",len(env.assigned_VNR_list))
                #print("sum:",vedge_sum)
                edge.append(vedge_sum/sedge_sum)
                node.append(vnode_sum/snode_sum)
                for j in range(3):
                    #fo=open(log[j],"a")
                    if j==0:
                        #fo.write(str.format("accept ratio on time {0}:{1}\n",int(env.time),success/i))
                        y0.append(success/i)
                    elif j==1:
                        #fo.write(str.format("revenue on time {0}:{1}\n",int(env.time),env.total_reward/(env.time+EPS)))
                        y1.append(env.total_reward/(env.time+EPS))
                    elif j==2:
                        #fo.write(str.format("cost ratio on time {0}:{1}\n",int(env.time),env.total_reward/(env.total_cost+EPS)))
                        y2.append(env.edge_reward/(env.total_cost+EPS))
        if(test_name=="rate"):
            acc_list.append(float(env.name.split("_")[1]))
        else:
            acc_list.append(int(env.name.split("_")[2]))
        x_list.append(x)
        y0_list.append(y0)
        y1_list.append(y1)
        y2_list.append(y2)
        edge_list.append(edge)
        node_list.append(node)
        name_list.append(env.name)
        suc_list.append(success/generated)
        fo=open("bbb","a")
        fo.write(str.format(env.name+" acc_ratio:{0}\n",success/generated))
        fo.close()

    title="different "+test_name+" request"
    fig1=plt.figure()
    fig1.suptitle("acceptance ratio - "+title)
    sub1=fig1.add_subplot(1,1,1)
    sub1.plot(acc_list,suc_list)
    fig1.savefig("acceptance_ratio_"+test_name+"_"+str(epoch)+".jpg")

    fig2=plt.figure()
    sub3=fig2.add_subplot(2,1,1)
    sub3.set_title("average revenue - "+title)
    for i in range(len(x_list)):
        sub3.plot(x_list[i],y1_list[i],label=name_list[i])
    sub3.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0)


    sub4=fig2.add_subplot(2,1,2)
    sub4.set_title("cost ratio - "+title)
    for i in range(len(x_list)):
        sub4.plot(x_list[i],y2_list[i],label=name_list[i])

    sub4.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0)
    fig2.tight_layout()
    fig2.savefig("revenue_cost_"+test_name+"_"+str(epoch)+".jpg")


    fig3=plt.figure()
    sub5=fig3.add_subplot(2,1,1)
    sub5.set_title("edge utility - "+title)
    for i in range(len(x_list)):
        sub5.plot(x_list[i],edge_list[i],label=name_list[i])
    sub5.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0)


    sub6=fig3.add_subplot(2,1,2)
    sub6.set_title("node utility - "+title)
    for i in range(len(x_list)):
        sub6.plot(x_list[i],node_list[i],label=name_list[i])

    sub6.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0)
    fig3.tight_layout()
    fig3.savefig("resource_ulitiliztion_"+test_name+"_"+str(epoch)+".jpg")


    fo=open("bbb","a")
    fo.write("\n")
    fo.close()




    return [test_name,x_list,y0_list,y1_list,y2_list,edge_list,node_list,name_list,acc_list,suc_list]



def test_env(env,generated,actor,sess,embedding="hybrid",type="static"):
    x=list()
    #x.append(0)
    y0=list()
    y1=list()
    y2=list()
    edge=list()
    node=list()
    success=0
    i=0

    while i < generated:
        # test_time=env.time
        # env.release_resource(test_time)
        s, v = env.get_state()
        # print("state acquired.")
        # snode_batch.append(s)
        # vnode_batch.append(v)
        env.snode_state = s
        env.vnode_state = v
        action_prob = actor.predict(s, v)
        # print("current action prob:",action_prob)
        # print(str.format("worker {0} current action prob:{1}",worker_index,action_prob))
        if(type=="static"):
            action_one_hot, action_pick = actor.pick_action(action_prob, env.substrate_network.attribute_list[2]["attributes"],
                                                        sess, phase="Testing")
        else:
            action_one_hot, action_pick = actor.random_pick_action()
        # print("action_pick:",action_pick)
        is_terminal, failure, reward = env.perform_action(action_pick)
        now = time.clock()

        if is_terminal == 1:
            success += 0
            # print(str.format("worker {0} failed.",worker_index))
        elif is_terminal == 2:
            success += 1
            # print(str.format("worker {0} success.",worker_index))
        # if(events==0):
        # step+=1
        # env.time=step
        if is_terminal != 0:
            i += 1
            env.VNR_counter += 1
            env.release_resource(env.time)
            x.append(int(env.time))
            for j in range(len(log)):
                fo=open(log[j],"a")
                if j==0:
                    #fo.write(str.format("accept ratio on time {0}:{1}\n",int(env.time),success/i))
                    y0.append(success/i)
                elif j==1:
                    #fo.write(str.format("revenue on time {0}:{1}\n",int(env.time),env.total_reward/(env.time+EPS)))
                    y1.append(env.total_reward/(env.time+EPS))
                elif j==2:
                    #fo.write(str.format("cost ratio on time {0}:{1}\n",int(env.time),env.total_reward/(env.total_cost+EPS)))
                    y2.append(env.total_reward/(env.total_cost+EPS))


    return [success,x,y0,y1,y2,env.name]




def master(network_parameter_queue, exp_queue):
    
    assert len(network_parameter_queue) == NUM_AGENT
    assert len(exp_queue) == NUM_AGENT
    logging.basicConfig(filename=LOG_FILE + '_master_meta',
                        filemode='w',
                        level=logging.INFO)
    epoch = 0


    '''c_graph=copy.deepcopy(G)
    c_graph_2=copy.deepcopy(G)
    c_graph_3=copy.deepcopy(G)
    c_graph_4=copy.deepcopy(G)
    c_graph_5=copy.deepcopy(G)
    c_graph_6=copy.deepcopy(G)
    c_env=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing")
    testing_list=c_env.generate_VNR_list(2000)
    c_env_2=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph_2,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing")
    c_env_3=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph_3,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",rate=0.08)
    testing_list_frequent=c_env_3.generate_VNR_list(4000)
    c_env_4=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph_4,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",VNR_type="edge_v")
    testing_list_edge=c_env_4.generate_VNR_list(2000)
    c_env_5=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph_5,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",VNR_type="node_v")
    testing_list_node=c_env_5.generate_VNR_list(2000)
    c_env_6=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph_6,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",VNR_type="intense")
    testing_list_intense=c_env_6.generate_VNR_list(2000)
    c_env_7=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph_6,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",VNR_type="very_intense")
    testing_list_very_intense=c_env_7.generate_VNR_list(2000)
    c_env_3_1=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph_3,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",rate=0.045)
    testing_list_frequent_1=c_env_3_1.generate_VNR_list(2250)
    c_env_3_2=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph_3,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",rate=0.05)
    testing_list_frequent_2=c_env_3_2.generate_VNR_list(2500)
    c_env_3_3=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph_3,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",rate=0.055)
    testing_list_frequent_3=c_env_3_3.generate_VNR_list(2750)
    c_env_3_4=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph_3,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",rate=0.06)
    testing_list_frequent_4=c_env_3_4.generate_VNR_list(3000)
    c_env_3_5=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph_3,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",rate=0.065)
    testing_list_frequent_5=c_env_3_5.generate_VNR_list(3250)
    c_env_3_6=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph_3,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",rate=0.07)
    testing_list_frequent_6=c_env_3_6.generate_VNR_list(3500)
    c_env_3_7=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph_3,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",rate=0.075)
    testing_list_frequent_7=c_env_3_7.generate_VNR_list(3750)
    c_env_4_1=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph_4,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",VNR_type="very_intense_edge")
    testing_list_edge_1=c_env_4_1.generate_VNR_list(2000)
    c_env_5_1=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph_5,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",VNR_type="very_intense_node")
    testing_list_node_1=c_env_5_1.generate_VNR_list(2000)
    c_env_6_1=en.Environment("Environment_Master", SNODE_SIZE,1000000,imported_graph=[c_graph_6,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",VNR_type="very_intense_all")
    testing_list_intense_1=c_env_6_1.generate_VNR_list(2000)
    c_env_noderank=en.Environment("noderank", SNODE_SIZE,1000000,imported_graph=[c_graph_3,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",rate=0.05,VNR_type="intense")
    testing_list_noderank=c_env_noderank.generate_VNR_list(2500)'''
    rate_list=rate_list_gen()
    edge_list,node_list,all_list=res_list_gen()
    size_list=size_list_gen()




    with tf.Session() as sess,open(LOG_FILE + '_master_0', 'w') as log_file:
        actor = ag.ActorNetwork(sess, "actor_master", INPUT_FEATURES, SNODE_SIZE, EXTRACTED_FEATURES, VNODE_FEATURES_SIZE,
                             ORDERS,laplacian=LAPLACIAN_TENSOR)
        critic = ag.CriticNetwork(sess, "critic_master", INPUT_FEATURES, SNODE_SIZE, EXTRACTED_FEATURES,
                               VNODE_FEATURES_SIZE, ORDERS,laplacian=LAPLACIAN_TENSOR)
        print(str("Network created."))
        #log_file.write(str("Network created.") + '\n')
        #log_file.flush()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print(str("Variables initialized."))
        #log_file.write(str("Variables initialized.") + '\n')
        #log_file.flush()
        writer=tf.summary.FileWriter(LOG_DIR+"/master",sess.graph)
        saver=tf.train.Saver(max_to_keep=10000)
        print(str("Saver created."))
        #log_file.write(str("Saver created.") + '\n')
        #log_file.flush()
        loaded_model=NN_MODEL
        if NN_MODEL is not None:
            saver.restore(sess,loaded_model)
            print("model loaded")

        sess.graph.finalize()
        while(epoch<5):
            start = time.clock()
            '''actor_parameters = actor.get_network_params()
            critic_parameters = critic.get_network_params()
            print(str("Network parameter get."))
            #log_file.write(str("Network parameter get.") + '\n')
            #log_file.flush()
            for i in range(NUM_AGENT):
                network_parameter_queue[i].put([actor_parameters, critic_parameters])
                #print(str.format("Network parameter put to worker {0}.",i))
                #log_file.write(str.format("Network parameter put to worker {0}.",i)+ '\n')
                #log_file.flush()
            actor_gradient_batch=[]
            critic_gradient_batch=[]
            for i in range(NUM_AGENT):
                s_batch, v_batch, a_batch, r_batch, terminal = exp_queue[i].get()
                print(str.format("Get exp from worker {0}.",i))
                # print("s_batch:",s_batch)
                # print("v_batch:", v_batch)
                # print("a_batch:", a_batch)
                # print("r_batch:", r_batch)
                #log_file.write(str.format("Get exp from worker {0}.",i)+ '\n')
                #log_file.flush()
                actor_gradient, critic_gradient, td_batch=ag.compute_gradients(s_batch=np.stack(s_batch,axis=0),vnode_batch=np.stack(v_batch,axis=0),a_batch=np.vstack(a_batch),r_batch=np.vstack(r_batch),terminal=terminal,actor=actor,critic=critic)
                #print(str.format("Gradient from worker {0}.",i))
                #log_file.write(str.format("Gradient from worker {0}.",i)+ '\n')
                #log_file.flush()
                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)


            assert  NUM_AGENT==len(actor_gradient_batch)
            #assert  len(actor_gradient_batch[0])==len(critic_gradient_batch[0])
            #for i in range(len(critic_gradient_batch[0])):
                #print("actor gradient batch:",actor_gradient_batch[i])
                #print("critic gradient batch:",critic_gradient_batch[i])
                #print("shape:",tf.shape(actor_gradient_batch[i]))
                # print("gradient length of master:",len(actor_gradient_batch[0][i]))
                # print("gradient of master:",actor_gradient_batch[0][i])
                # for j in range (len(actor_gradient_batch[i])):
                #     print("gradient shape of element %d:"%j, np.shape(actor_gradient_batch[i][j]))
                #actor.apply_gradients(actor_gradient_batch[0][i])
                #critic.apply_gradients(critic_gradient_batch[0][i])'''

            now = time.clock()
            print(str.format("training step {0} costs:{1}",epoch,now - start))
            #log_file.write(str.format("training step {0} costs:{1}",epoch,now - start)+'\n')
            #log_file.flush()
            epoch+=1
            if(epoch%1000==0):
                save=saver.save(sess,NEW_DIR+"_ep_"+str(epoch)+".ckpt")
                print("Model saved in file:"+save)
            if(epoch<=5):
                '''master_graph=copy.deepcopy(G)
                master_graph_2=copy.deepcopy(G)
                master_graph_3=copy.deepcopy(G)
                master_graph_5=copy.deepcopy(G)
                master_graph_4=copy.deepcopy(G)
                master_graph_6=copy.deepcopy(G)
                master_graph_7=copy.deepcopy(G)



                master_env=en.Environment("mini", SNODE_SIZE,1000000,imported_graph=[master_graph,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing")
                master_env.testing_VNR_list=copy.deepcopy(testing_list)
                
                master_env_2=en.Environment("mini_random", SNODE_SIZE,1000000,imported_graph=[master_graph_2,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing")
                master_env_2.testing_VNR_list=copy.deepcopy(testing_list)
                   
                master_env_3=en.Environment("rate_0.08", SNODE_SIZE,1000000,imported_graph=[master_graph_3,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",rate=0.08)
                master_env_3.testing_VNR_list=copy.deepcopy(testing_list_frequent)
                master_env_4=en.Environment("edge_50", SNODE_SIZE,1000000,imported_graph=[master_graph_4,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",VNR_type="edge_v")
                master_env_4.testing_VNR_list=copy.deepcopy(testing_list_edge)
                master_env_5=en.Environment("node_50", SNODE_SIZE,1000000,imported_graph=[master_graph_5,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",VNR_type="node_v")
                master_env_5.testing_VNR_list=copy.deepcopy(testing_list_node)

                master_env_6=en.Environment("all_50", SNODE_SIZE,1000000,imported_graph=[master_graph_6,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",VNR_type="intense")
                master_env_6.testing_VNR_list=copy.deepcopy(testing_list_intense)
                master_env_7=en.Environment("all_100", SNODE_SIZE,1000000,imported_graph=[master_graph_7,node_attr,wmin,wmax],link_embedding_type="hybrid",phase="Testing",VNR_type="very_intense")
                master_env_7.testing_VNR_list=copy.deepcopy(testing_list_very_intense)

                master_graph_3_1 = copy.deepcopy(G)
                master_graph_3_2 = copy.deepcopy(G)
                master_graph_3_3 = copy.deepcopy(G)
                master_graph_3_4 = copy.deepcopy(G)
                master_graph_3_5 = copy.deepcopy(G)
                master_graph_3_6 = copy.deepcopy(G)
                master_graph_3_7 = copy.deepcopy(G)


                master_env_3_1 = en.Environment("rate_0.045", SNODE_SIZE, 1000000,
                                                imported_graph=[master_graph_3_1, node_attr, wmin, wmax],
                                                link_embedding_type="hybrid",
                                                phase="Testing", rate=0.045)
                master_env_3_1.testing_VNR_list = copy.deepcopy(testing_list_frequent_1)
                master_env_3_2 = en.Environment("rate_0.05", SNODE_SIZE, 1000000,
                                                imported_graph=[master_graph_3_2, node_attr, wmin, wmax],
                                                link_embedding_type="hybrid",
                                                phase="Testing", rate=0.05)
                master_env_3_2.testing_VNR_list = copy.deepcopy(testing_list_frequent_2)
                master_env_3_3 = en.Environment("rate_0.055", SNODE_SIZE, 1000000,
                                                imported_graph=[master_graph_3_3, node_attr, wmin, wmax],
                                                link_embedding_type="hybrid",
                                                phase="Testing", rate=0.055)
                master_env_3_3.testing_VNR_list = copy.deepcopy(testing_list_frequent_3)
                master_env_3_4 = en.Environment("rate_0.06", SNODE_SIZE, 1000000,
                                                imported_graph=[master_graph_3_4, node_attr, wmin, wmax],
                                                link_embedding_type="hybrid",
                                                phase="Testing", rate=0.05)
                master_env_3_4.testing_VNR_list = copy.deepcopy(testing_list_frequent_4)
                master_env_3_5 = en.Environment("rate_0.065", SNODE_SIZE, 1000000,
                                                imported_graph=[master_graph_3_5, node_attr, wmin, wmax],
                                                link_embedding_type="hybrid",
                                                phase="Testing", rate=0.065)
                master_env_3_5.testing_VNR_list = copy.deepcopy(testing_list_frequent_5)
                master_env_3_6 = en.Environment("rate_0.07", SNODE_SIZE, 1000000,
                                                imported_graph=[master_graph_3_6, node_attr, wmin, wmax],
                                                link_embedding_type="hybrid",
                                                phase="Testing", rate=0.07)
                master_env_3_6.testing_VNR_list = copy.deepcopy(testing_list_frequent_6)
                master_env_3_7 = en.Environment("rate_0.075", SNODE_SIZE, 1000000,
                                                imported_graph=[master_graph_3_7, node_attr, wmin, wmax],
                                                link_embedding_type="hybrid",
                                                phase="Testing", rate=0.075)
                master_env_3_7.testing_VNR_list = copy.deepcopy(testing_list_frequent_7)




                master_graph_4_1 = copy.deepcopy(G)
                master_graph_5_1 = copy.deepcopy(G)
                master_graph_6_1 = copy.deepcopy(G)


                master_env_4_1 = en.Environment("edge_70", SNODE_SIZE, 1000000,
                                                imported_graph=[master_graph_4_1, node_attr, wmin, wmax],
                                                link_embedding_type="hybrid",
                                                phase="Testing")
                master_env_4_1.testing_VNR_list = copy.deepcopy(testing_list_edge_1)


                master_env_5_1 = en.Environment("node_70", SNODE_SIZE, 1000000,
                                                imported_graph=[master_graph_5_1, node_attr, wmin, wmax],
                                                link_embedding_type="hybrid",
                                                phase="Testing")
                master_env_5_1.testing_VNR_list = copy.deepcopy(testing_list_node_1)
                
                master_env_6_1 = en.Environment("all_70", SNODE_SIZE, 1000000,
                                                imported_graph=[master_graph_6_1, node_attr, wmin, wmax],
                                                link_embedding_type="hybrid",
                                                phase="Testing")
                master_env_6_1.testing_VNR_list = copy.deepcopy(testing_list_intense_1)
                master_graph_noderank = copy.deepcopy(G)

                master_env_noderank = en.Environment("noderank",SNODE_SIZE, 1000000,
                                imported_graph=[master_graph_noderank, node_attr, wmin, wmax],
                                link_embedding_type="hybrid",
                                phase="Testing")


                master_env_noderank.testing_VNR_list = copy.deepcopy(testing_list_noderank)

                print("Start Testing:")
                success=test_env(master_env,2000,actor,sess)
                success_r=test_env(master_env_2,2000,actor,sess,type="random")
                success_f=test_env(master_env_3,4000,actor,sess)
                success_e=test_env(master_env_4,2000,actor,sess)
                success_n=test_env(master_env_5,2000,actor,sess)
                success_i=test_env(master_env_6,2000,actor,sess)
                success_f_1 = test_env(master_env_3_1, 2250, actor, sess)
                success_f_2 = test_env(master_env_3_2, 2500, actor, sess)
                success_f_3 = test_env(master_env_3_3, 2750, actor, sess)
                success_f_4 = test_env(master_env_3_4, 3000, actor, sess)
                success_f_5 = test_env(master_env_3_5, 3250, actor, sess)
                success_f_6 = test_env(master_env_3_6, 3500, actor, sess)
                success_f_7 = test_env(master_env_3_7, 3750, actor, sess)

                success_e_1 = test_env(master_env_4_1, 2000, actor, sess)
                success_n_1 = test_env(master_env_5_1, 2000, actor, sess)
                success_i_1 = test_env(master_env_6_1, 2000, actor, sess)
                success_vi=test_env(master_env_7,2000,actor,sess)
                success_noderank=test_env(master_env_noderank,2500,actor,sess)


                fig1=plt.figure()
                sub1=fig1.add_subplot(2,1,1)
                sub1.set_title("acceptance ratio - various resource demand")
                sub1.plot(success[1],success[2],label=success[5])
                #sub1.plot(success_r[1],success_r[2],label=success_r[5])
                sub1.plot(success_e[1],success_e[2],label=success_e[5])
                sub1.plot(success_n[1],success_n[2],label=success_n[5])
                sub1.plot(success_i[1],success_i[2],label=success_i[5])
                sub1.plot(success_e_1[1],success_e_1[2],label=success_e_1[5])
                sub1.plot(success_n_1[1],success_n_1[2],label=success_n_1[5])
                sub1.plot(success_i_1[1],success_i_1[2],label=success_i_1[5])
                sub1.plot(success_vi[1],success_vi[2],label=success_vi[5])
                sub1.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0)

                sub2=fig1.add_subplot(2,1,2)
                sub2.set_title("acceptance ratio - various request arriving rate")
                sub2.plot(success[1],success[2],label=success[5])
                sub2.plot(success_f_1[1],success_f_1[2],label=success_f_1[5])
                sub2.plot(success_f_2[1],success_f_2[2],label=success_f_2[5])
                sub2.plot(success_f_3[1],success_f_3[2],label=success_f_3[5])
                sub2.plot(success_f_4[1],success_f_4[2],label=success_f_4[5])
                sub2.plot(success_f_5[1],success_f_5[2],label=success_f_5[5])
                sub2.plot(success_f_6[1],success_f_6[2],label=success_f_6[5])
                sub2.plot(success_f_7[1],success_f_7[2],label=success_f_7[5])
                sub2.plot(success_f[1],success_f[2],label=success_f[5])
                sub2.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0)

                fig1.tight_layout()
                fig1.savefig("acceptance_ratio_"+str(epoch)+".jpg")

                fig2=plt.figure()
                sub3=fig2.add_subplot(2,1,1)
                sub3.set_title("average revenue - various resource demand")
                sub3.plot(success[1],success[3],label=success[5])
                #sub3.plot(success_r[1],success_r[3],label=success_r[5])
                sub3.plot(success_e[1],success_e[3],label=success_e[5])
                sub3.plot(success_n[1],success_n[3],label=success_n[5])
                sub3.plot(success_i[1],success_i[3],label=success_i[5])
                sub3.plot(success_e_1[1],success_e_1[3],label=success_e_1[5])
                sub3.plot(success_n_1[1],success_n_1[3],label=success_n_1[5])
                sub3.plot(success_i_1[1],success_i_1[3],label=success_i_1[5])
                sub3.plot(success_vi[1],success_vi[3],label=success_vi[5])
                sub3.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0)


                sub4=fig2.add_subplot(2,1,2)
                sub4.set_title("average revenue - various request arriving rate")
                sub4.plot(success[1],success[3],label=success[5])
                sub4.plot(success_f_1[1],success_f_1[3],label=success_f_1[5])
                sub4.plot(success_f_2[1],success_f_2[3],label=success_f_2[5])
                sub4.plot(success_f_3[1],success_f_3[3],label=success_f_3[5])
                sub4.plot(success_f_4[1],success_f_4[3],label=success_f_4[5])
                sub4.plot(success_f_5[1],success_f_5[3],label=success_f_5[5])
                sub4.plot(success_f_6[1],success_f_6[3],label=success_f_6[5])
                sub4.plot(success_f_7[1],success_f_7[3],label=success_f_7[5])
                sub4.plot(success_f[1],success_f[3],label=success_f[5])

                sub4.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0)
                fig2.tight_layout()
                fig2.savefig("average_revenue_"+str(epoch)+".jpg")

                fig3=plt.figure()
                sub5=fig3.add_subplot(2,1,1)
                sub5.set_title("cost ratio - various resource demand")
                sub5.plot(success[1],success[3],label=success[5])
                #sub5.plot(success_r[1],success_r[3],label=success_r[5])
                sub5.plot(success_e[1],success_e[3],label=success_e[5])
                sub5.plot(success_n[1],success_n[3],label=success_n[5])
                sub5.plot(success_i[1],success_i[3],label=success_i[5])
                sub5.plot(success_e_1[1],success_e_1[3],label=success_e_1[5])
                sub5.plot(success_n_1[1],success_n_1[3],label=success_n_1[5])
                sub5.plot(success_i_1[1],success_i_1[3],label=success_i_1[5])
                sub5.plot(success_vi[1],success_vi[3],label=success_vi[5])
                sub5.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0)


                sub6=fig3.add_subplot(2,1,2)
                sub6.set_title("cost ratio - various request arriving rate")
                sub6.plot(success[1],success[3],label=success[5])
                sub6.plot(success_f_1[1],success_f_1[3],label=success_f_1[5])
                sub6.plot(success_f_2[1],success_f_2[3],label=success_f_2[5])
                sub6.plot(success_f_3[1],success_f_3[3],label=success_f_3[5])
                sub6.plot(success_f_4[1],success_f_4[3],label=success_f_4[5])
                sub6.plot(success_f_5[1],success_f_5[3],label=success_f_5[5])
                sub6.plot(success_f_6[1],success_f_6[3],label=success_f_6[5])
                sub6.plot(success_f_7[1],success_f_7[3],label=success_f_7[5])
                sub6.plot(success_f[1],success_f[3],label=success_f[5])
                sub6.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0)


                fig3.tight_layout()
                fig3.savefig("cost_ratio_"+str(epoch)+".jpg")

  

                    
                print("current acc ratio:",success[0]/2000)
                fo=open("bbb","a")
                fo.write(str.format("static acc_ratio:{0}\n",success[0]/2000))
                fo.write(str.format("static ran_ratio:{0}\n",success_r[0]/2000))
                fo.write(str.format("freq_0.045_ratio:{0}\n",success_f_1[0]/2250))
                fo.write(str.format("freq_0.05_ratio:{0}\n", success_f_2[0] / 2500))
                fo.write(str.format("freq_0.055_ratio:{0}\n", success_f_3[0] / 2750))
                fo.write(str.format("freq_0.06_ratio:{0}\n", success_f_4[0] / 3000))
                fo.write(str.format("freq_0.065_ratio:{0}\n", success_f_5[0] / 3250))
                fo.write(str.format("freq_0.07_ratio:{0}\n", success_f_6[0] / 3500))
                fo.write(str.format("freq_0.075_ratio:{0}\n", success_f_7[0] / 3750))
                fo.write(str.format("freq_0.08_ratio:{0}\n",success_f[0]/4000))
                fo.write(str.format("edge_ratio:{0}\n",success_e[0]/2000))
                fo.write(str.format("node_ratio:{0}\n",success_n[0]/2000))
                fo.write(str.format("intense_ratio:{0}\n",success_i[0]/2000))
                fo.write(str.format("edge_up_ratio:{0}\n", success_e_1[0] / 2000))
                fo.write(str.format("node_up_ratio:{0}\n", success_n_1[0] / 2000))
                fo.write(str.format("intense_up_ratio:{0}\n", success_i_1[0] / 2000))
                fo.write(str.format("very_intense_ratio:{0}\n", success_vi[0] / 2000))
                fo.write(str.format("noderank_ratio:{0}\n", success_noderank[0] / 2500))
                fo.write("\n")
                fo.close()'''

                res_rate=test_env_list("rate",actor,sess,rate_list,epoch=epoch)
                res_edge=test_env_list("edge",actor,sess,edge_list,epoch=epoch)
                res_node=test_env_list("node",actor,sess,node_list,epoch=epoch)
                res_all=test_env_list("all",actor,sess,all_list,epoch=epoch)
                res_size=test_env_list("size",actor,sess,size_list,epoch=epoch)
                pickle.dump(res_rate,open(NEW_DIR+"res_rate_"+str(epoch)+".var",'wb'))
                pickle.dump(res_edge,open(NEW_DIR+"res_edge_"+str(epoch)+".var",'wb'))
                pickle.dump(res_node,open(NEW_DIR+"res_node_"+str(epoch)+".var",'wb'))
                pickle.dump(res_all,open(NEW_DIR+"res_all_"+str(epoch)+".var",'wb'))
                pickle.dump(res_size,open(NEW_DIR+"res_size_"+str(epoch)+".var",'wb'))

                fig1=plt.figure()
                fig1.suptitle("acceptance ratio - various resource request")
                sub1=fig1.add_subplot(1,1,1)
                sub1.plot(res_edge[8],res_edge[9],label=res_edge[0])
                sub1.plot(res_node[8],res_node[9],label=res_node[0])
                sub1.plot(res_all[8],res_all[9],label=res_all[0])
                sub1.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0)
                fig1.savefig("acceptance_ratio_various"+"_"+str(epoch)+".jpg")


                #if(success/2000>0.67):
                    #save=saver.save(sess,GOOD_DIR+"ep_"+str(epoch)+"_acc_"+str(success/2000)+".ckpt")            



                    
                
            ##tf summary need to be done







def worker(worker_index,G,wmin,wmax,network_parameter_queue, exp_queue):
    # assert len(network_parameter_queue) == NUM_AGENT
    # assert len(exp_queue) == NUM_AGENT
    np.random.seed(RANDOM_SEED+worker_index)
    step=0
    env_graph=copy.deepcopy(G)
    env = en.Environment(str.format("Environment_{0}",worker_index), SNODE_SIZE,1000000,imported_graph=[env_graph,node_attr,wmin,wmax],link_embedding_type="hybrid")
    pickle.dump(env, open(str.format('env_worker_{0}.var',worker_index), 'wb'))
    with tf.Session() as sess,open(LOG_FILE + 'worker' + str(worker_index), 'w') as log_file:
        actor = ag.ActorNetwork(sess, str.format("actor_worker_{0}",worker_index), INPUT_FEATURES, SNODE_SIZE, EXTRACTED_FEATURES, VNODE_FEATURES_SIZE,
                             ORDERS,laplacian=LAPLACIAN_TENSOR)
        critic = ag.CriticNetwork(sess, str.format("actor_worker_{0}",worker_index), INPUT_FEATURES, SNODE_SIZE, EXTRACTED_FEATURES,
                               VNODE_FEATURES_SIZE, ORDERS,laplacian=LAPLACIAN_TENSOR)
        actor_parameters, critic_parameters = network_parameter_queue.get()
        actor.set_network_params(actor_parameters)
        critic.set_network_params(critic_parameters)
        #print("worker network parameter:", actor_parameters)

        state = env.get_state()
        # action
        snode_batch = []
        vnode_batch=[]
        a_batch = []
        r_batch = []
        
        sess.graph.finalize()
        ass=0
        while (True):
            start = time.clock()
            s,v = env.get_state()
            snode_batch.append(s)
            vnode_batch.append(v)
            env.snode_state=s
            env.vnode_state=v
            action_prob = actor.predict(s, v)
            out_s,out_v,out_b=actor.out_debug(s,v)
            print(str.format("worker {0} current action prob:{1}",worker_index,action_prob))
            print("out_s:",out_s)
            print("out_v:",out_v)
            print("out_b:",out_b)
            action_one_hot,action_pick=actor.pick_action(action_prob,env.substrate_network.attribute_list[2]["attributes"],sess)
            is_terminal, failure, reward = env.perform_action(action_pick)
            '''if(action_pick!=0):
                reward=-100
            else:
                reward=100'''
            print("action:",action_pick)
            ass+=1
            a_batch.append(action_one_hot)
            r_batch.append(reward)

            if(is_terminal!=0):
                ass=0
                s,v = env.get_state()
                snode_batch.append(s)
                vnode_batch.append(v)
                exp_queue.put([snode_batch,vnode_batch,a_batch,r_batch,is_terminal])
                actor_parameters,critic_parameters=network_parameter_queue.get()
                #print("worker network parameter shape:",tf.shape(actor_parameters))
                actor.set_network_params(actor_parameters)
                critic.set_network_params(critic_parameters)

                del snode_batch[:]
                del vnode_batch[:]
                del a_batch[:]
                del r_batch[:]
            now = time.clock()
            print(str.format("worker {0} sampling step {1} costs:{2}",worker_index, step, now - start))
            #log_file.write(str.format("worker {0} sampling step {1} costs:{2}",worker_index, step, now - start) + '\n')
            #log_file.flush()
            if is_terminal==1:
                #print(str.format("worker {0} failed.",worker_index))
                step +=3
            elif is_terminal==2:
                #print(str.format("worker {0} success.",worker_index))
                step+=1
            env.time=step
            if is_terminal!=0:
                env.release_resource(env.time)
                #print(str.format("worker {0} now assigned cpu list:{1}", worker_index,
                #             env.current_assigned_cpu))
                #print(str.format("worker {0} now with CPU in use:{1}",worker_index, env.substrate_network.attribute_list[0]["attributes"]))
                #print(str.format("worker {0} now with bandwidth in use:{1}", worker_index,
                #            env.substrate_network.attribute_list[1]["attributes"]))
                #print(str.format("worker {0} now assigned vnr list:{1}", worker_index,
                #             len(env.assigned_VNR_list)))

#np.random.seed(RANDOM_SEED)
network_parameter_queue=[]
exp_queue=[]
for i in range(NUM_AGENT):
    network_parameter_queue.append(mp.Queue(1))
    exp_queue.append(mp.Queue(1))

coordinator=mp.Process(target=master,args=(network_parameter_queue,exp_queue))
coordinator.start()
workers=[]
for i in range(NUM_AGENT):
    workers.append(mp.Process(target=worker,args=(i,G,wmin,wmax,network_parameter_queue[i],exp_queue[i])))

for i in range(NUM_AGENT):
    workers[i].start()

coordinator.join()



    # now = time.clock()
    # print(now - start)
    # start = time.clock()
    # for i in range(200):
    #     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #     run_metadata=tf.RunMetadata()
    #     s, v = env.get_state()
    #     td=cc.predict(s,v)
    #     action=aa.predict(s,v)
    #     print("the action prob is:",action)
    #     action_one_hot,action_pick=aa.pick_action(action,env.substrate_network.attribute_list[4]["attributes"])
    #     # print("this is training feed:", action_pick, td)
    #     is_terminal,failure,reward=env.perform_action(action_pick)
    #     print(str("this is after perform action:"),is_terminal,failure,reward)
    #     # print()
    #     s_,v_=env.get_state()
    #     td_=cc.predict(s_,v_)
    #     print(str("td initialized from critic:"), td)
    #     print(str("td acted from next critic:"), td_)
    #     td_error=td_target(td_,reward)
    #
    #     new_td=cc.get_td(s,v,td_error)
    #     print(str("td created from reward:"),new_td)
    #     c_optimize,c_loss_summary,c_summary=cc.train(s, v, td_error)
    #     #a=aa.train(s,v,LAPLACIAN_TENSOR,action_one_hot,td)
    #     a_objective,a_entropy,a_entropy_summary,a_optimize,a_summary=aa.train(s,v,action_one_hot,new_td)
    #     print(str.format("actor objective:{0},{1}",a_objective,a_entropy))
    #     now = time.clock()
    #     print(now - start)
    #     start = time.clock()
    #     ENTROPY_WEIGHT=ENTROPY_WEIGHT/1.01
    #     print("EntropyWeight:%.6f" % ENTROPY_WEIGHT)
    #     print(now - start)
    #     start = time.clock()
    #     ENTROPY_WEIGHT=ENTROPY_WEIGHT/1.01
    #     print("EntropyWeight:%.6f" % ENTROPY_WEIGHT)
    #     train_writer.add_run_metadata(run_metadata,'step%03d'%i)
    #     train_writer.add_summary(c_summary,i)
