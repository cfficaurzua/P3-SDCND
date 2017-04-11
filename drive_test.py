def retrieve_steering_angle(neurons, width = (-0.75,0.75)):
    neurons = neurons[0,:]
    n = neurons.shape[0]
    means = np.linspace(width[0],width[1], n)
    plt.bar(means,neurons, width = 0.1, align='center')
    output = np.dot(means,neurons)/np.sum(neurons)
    return output
