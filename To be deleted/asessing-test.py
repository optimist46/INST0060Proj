#initialise variables
#this is just a code snippet not working alone

correct = 0
correct_classes = [0 for i in range(len(classes))]
guess_classes = [0 for i in range(len(classes))]

#run through the predictions

for i in range(len(inputs_test[:,0])):
    pred = int(reg.predict(inputs_test[i].reshape(1, -1)))
    actual = int(targets_test[i])
    guess_classes[pred] += 1
    #print(pred-actual)
    if pred-actual==0:
        correct+=1
        correct_classes[pred]+=1

#print asessment

for i in range(len(classes)):
    print(f"{classes[i]}: {guess_classes[i]}/300 accuracy: {correct_classes[i]/len(inputs_test[:,0])*len(classes)*100:.3f}%")
    
print(f"----------------")
print(f"{correct/len(inputs_test[:,0])*100:.3f}%")