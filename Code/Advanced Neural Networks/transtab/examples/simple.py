import os
os.chdir('../')

import transtab
from sklearn.model_selection import KFold
# set random seed
transtab.random_seed(42)
import numpy as np
#%%
# load a dataset and start vanilla supervised training
# allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = transtab.load_data(['credit-g', 'credit-approval'])
# for i in range(8):
#     classflag=i

def pre(classflag):
    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = transtab.load_data('data',classflag)
    X,y=allset
    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    acc, f1, rec,pre,auc=[],[],[],[],[]
    mae=[]
    mse=[]

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print(f'KFold {i + 1}:')
        # print("Train index:", train_index, "Test index:", test_index)
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # print("Train y:", y_train, "Test y:", y_test)
        trainset=(x_train,y_train)
        valset=(x_test,y_test)
        testset=valset
    # build transtab classifier model
        if classflag==0 or classflag==1 or classflag==5 :
            model = transtab.build_classifier(cat_cols, num_cols, bin_cols,num_class=4)
        elif classflag==3 or classflag==4:
            model = transtab.build_classifier(cat_cols, num_cols, bin_cols, num_class=2)
        elif classflag==2 or classflag==6 or classflag==7:
            model = transtab.build_classifier(cat_cols, num_cols, bin_cols, num_class=3)



        # #
        # start training
        training_arguments = {
            'num_epoch':10,
            'eval_metric':'val_loss',
            'eval_less_is_better':True,
            'output_dir':'./checkpoint',
            'batch_size':128,
            'lr':1e-4,
            'weight_decay':1e-4,
            }
        transtab.train(model, trainset, valset, **training_arguments)

        # save model
        model.save(r'./ckpt/pretrained_{}'.format(classflag))

        # model.load('./ckpt/pretrained')

        x_test, y_test = testset
        ypred = transtab.predict(model, x_test,y_test)
        #%%
        # evaluate the predictions with bootstrapping estimate
        if classflag==0:
            res_acc=transtab.evaluate(ypred, y_test, seed=42, metric='acc')
            res_auc=transtab.evaluate(ypred, y_test, seed=42, metric='auc')
            res_pre = transtab.evaluate(ypred, y_test, seed=42, metric='pre')
            res_rec = transtab.evaluate(ypred, y_test, seed=42, metric='rec')
            res_f1 = transtab.evaluate(ypred, y_test, seed=42, metric='f1')
            acc.append(res_acc)
            auc.append(res_auc)
            pre.append(res_pre)
            rec.append(res_rec)
            f1.append(res_f1)
            print("classification:", classflag)
            print("fold=",i)
            print("res_acc=", res_acc)
            print("res_auc=", res_auc)
            print("res_pre=", res_pre)
            print("res_rec=", res_rec)
            print("res_f1=", res_f1)


        else:
            res_mse = transtab.evaluate(ypred, y_test, seed=42, metric='mse')
            res_mae = transtab.evaluate(ypred, y_test, seed=42, metric='mae')
            mae.append(res_mae)
            mse.append(res_mse)
            print("regression:", classflag)
            print("fold=", i)
            print("res_mse=", res_mse)
            print("res_mae=", res_mae)


    print("classification:",classflag)
    print("res_acc=", acc, np.mean(acc))
    print("res_auc=", auc,np.mean(auc))
    print("res_pre=", pre, np.mean(pre))
    print("res_rec=", rec, np.mean(rec))
    print("res_f1=", f1,np.mean(f1))
    print("regression:", classflag)
    print("res_mse=", mse, np.mean(mse))
    print("res_mae=", mae, np.mean(mae))

for classflag in range(4,5):
    pre(classflag)