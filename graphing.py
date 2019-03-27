import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score


def graph_2dim_results(A, B, dataset, labels, dataset_infected, inf_points, colors,  svc_orig, svc_inf):
    step = (B - A) / 100.0  # step size in the mesh
    x_min, x_max = -0.2, 1.2
    y_min, y_max = -0.2, 1.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    plt.subplot(221)
    plt.title('original dataset')
    plt.scatter([float(i[0]) for i in dataset], [float(i[1]) for i in dataset], c=colors, cmap=plt.cm.coolwarm)
    plt.subplot(222)
    plt.title('infected dataset')
    plt.scatter([float(i[0]) for i in dataset_infected], [float(i[1]) for i in dataset_infected], c=colors,
                cmap=plt.cm.coolwarm)
    plt.plot([i[0] for i in inf_points], [i[1] for i in inf_points], 'go', mfc='none')

    predicted_labels_inf_svc = svc_inf.predict(dataset)
    err_inf_svc = 1 - accuracy_score(labels, predicted_labels_inf_svc)

    plt.subplot(224)
    Z = svc_inf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter([float(i[0]) for i in dataset], [float(i[1]) for i in dataset], c=colors, cmap=plt.cm.coolwarm)
    plt.title('inf svc on orig data, err='+str(int(100*err_inf_svc))+'%')

    predicted_labels_orig_svc = svc_orig.predict(dataset)
    err_orig_svc = 1 - accuracy_score(labels, predicted_labels_orig_svc)

    plt.subplot(223)
    Z = svc_orig.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter([float(i[0]) for i in dataset], [float(i[1]) for i in dataset], c=colors, cmap=plt.cm.coolwarm)
    plt.title('orig svc on orig data, err='+str(int(100*err_orig_svc))+'%')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()


def graph_multidim_results(test_errs, classifier_names, n):
    time_pts = np.arange(1, n)

    fig, ax = plt.subplots()
    n_c = len(classifier_names)

    for i in range(n_c):
        ax.plot(time_pts, test_errs[i, :], '-', label=classifier_names[i], alpha=0.7)

    plt.ylim(top=0.5)
    legend = ax.legend(loc='upper left', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    plt.show()