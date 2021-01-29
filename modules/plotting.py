from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class PlottingModule():
  def plotcol(self, df_training, colname, wrong_indexes_0, wrong_indexes_1):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.hist(df_training[colname])
    ax2.hist(df_training.iloc[wrong_indexes_0, :][colname])
    ax3.hist(df_training.iloc[wrong_indexes_1, :][colname])
    ax1.set_title(colname)
    ax2.set_title('FP')
    ax3.set_title('FN')
    plt.show()

  def calc_specificity(self, y_actual, y_pred, thresh):
      # calculates specificity
      return sum((y_pred < thresh) & (y_actual == 0)) /sum(y_actual ==0)

  def calc_prevalence(self, y_actual):
      # calculates prevalence
      return sum((y_actual == 1)) /len(y_actual)

  def show_curve(self, df_train_pred, df_val_pred, df_test_pred, y_train, y_val, y_test, threshold=0.5):
    print('Accuracy', accuracy_score(y_test, df_test_pred['predicted_value']))
    print('Area under the curve', roc_auc_score(y_test, df_test_pred['predicted_value']))

    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, df_train_pred['predicted_value'])
    fpr_val, tpr_val, thresholds_val = roc_curve(y_val, df_val_pred['predicted_value'])
    auc_train = roc_auc_score(y_train, df_train_pred['predicted_value'])
    auc_val = roc_auc_score(y_val, df_val_pred['predicted_value'])

    print('Train prevalence:%.3f'%calc_prevalence(y_train))
    print('Valid prevalence:%.3f'%calc_prevalence(y_val))

    print('Train AUC:%.3f'%auc_train)
    print('Valid AUC:%.3f'%auc_val)

    print('Train accuracy:%.3f'%accuracy_score(y_train, df_train_pred['predicted_value'] >= threshold))
    print('Valid accuracy:%.3f'%accuracy_score(y_val, df_val_pred['predicted_value'] >= threshold))

    print('Train recall:%.3f'%recall_score(y_train, df_train_pred['predicted_value']>= threshold))
    print('Valid recall:%.3f'%recall_score(y_val, df_val_pred['predicted_value']>= threshold))

    print('Train precision:%.3f'%precision_score(y_train, df_train_pred['predicted_value']>= threshold))
    print('Valid precision:%.3f'%precision_score(y_val, df_val_pred['predicted_value']>= threshold))

    print('Train specificity:%.3f'%calc_specificity(y_train, df_train_pred['predicted_value'], threshold))
    print('Valid specificity:%.3f'%calc_specificity(y_val, df_val_pred['predicted_value'], threshold))

    plt.plot(fpr_train, tpr_train,'r-', label = 'Train AUC: %.2f'%auc_train)
    plt.plot(fpr_val, tpr_val,'b-',label = 'Valid AUC: %.2f'%auc_val)
    plt.plot([0,1],[0,1],'-k') # True Positive Rate == False Positive Rate
    # False Positive Rate = proportion of not diabetes samples that where incorrectly classified = (1-Specificity) 
    plt.xlabel('False Positive Rate = (1-Specificity):',fontsize = 12) 
    # True Positive Rate = proportion of diabetes samples that where correctly classified = Sensitivity
    plt.ylabel('True Positive Rate = (Sensitivity):',fontsize = 12) 
    plt.legend(fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.show()

    cm = confusion_matrix(y_test, df_test_pred['predicted_value'])
    print('Confusion matrix\n\n', cm)
    print('\nTrue Positives(TP) = ', cm[0,0])
    print('\nTrue Negatives(TN) = ', cm[1,1])
    print('\nFalse Positives(FP) = ', cm[0,1])
    print('\nFalse Negatives(FN) = ', cm[1,0])
