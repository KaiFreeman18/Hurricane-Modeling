# Formula for determining longitude and latitude as well as creating a window for storm potential
def FeatureColumnsXYZ(dframe):
    dframe['x'] = np.cos(np.radians(dframe.Lat)) * np.cos(np.radians(dframe.Lon))
    dframe['y'] = np.cos(np.radians(dframe.Lat)) * np.sin(np.radians(dframe.Lon))
    dframe['z'] = np.sin(np.radians(dframe.Lat))
    
    dframe['xLag1'] = np.cos(np.radians(dframe.LatLag1)) * np.cos(np.radians(dframe.LonLag1))
    dframe['yLag1'] = np.cos(np.radians(dframe.LatLag1)) * np.sin(np.radians(dframe.LonLag1))
    dframe['zLag1'] = np.sin(np.radians(dframe.LatLag1))
    
# Seperating data into managable columns
def FeatureColumns(dframe):
    grid1 = [(-70.0,5.0),(-70.0,20.0),(-10.0,20.0),(-10.0,5.0),(-70.0,5.0)]
    grid2 = [(-100.0,5.0),(-100.0,20.0),(-70.0,20.0),(-70.0,5.0),(-100.0,5.0)]
    grid3 = [(-70.0,20.0),(-70.0,40.0),(-10.0,40.0),(-10.0,20.0),(-70.0,20.0)]
    grid4 = [(-100.0,20.0),(-100.0,40.0),(-70.0,40.0),(-70.0,20.0),(-100.0,20.0)]
    poly1 = mpltPath.Path(grid1)
    poly2 = mpltPath.Path(grid2)
    poly3 = mpltPath.Path(grid3)
    poly4 = mpltPath.Path(grid4)
    
    Genesis = dframe.groupby('StormID').first()
    Genesis['GroupID'] = 0
    
    for row in range(0,Genesis.shape[0]):
        point = ([Genesis.Lon[row], Genesis.Lat[row]])
        inside1 = poly1.contains_point(point)
        inside2 = poly2.contains_point(point)
        inside3 = poly3.contains_point(point)
        inside4 = poly4.contains_point(point)
        if inside1:
            Genesis.GroupID.iloc[row] = 1
        if inside2:       
            Genesis.GroupID.iloc[row] = 2
        if inside3:
            Genesis.GroupID.iloc[row] = 3
        elif inside4:
            Genesis.GroupID.iloc[row] = 4
    
    train_events_g1 = Genesis[Genesis.GroupID == 1].index.values.tolist()
    train_events_g2 = Genesis[Genesis.GroupID == 2].index.values.tolist()
    train_events_g3 = Genesis[Genesis.GroupID == 3].index.values.tolist()
    train_events_g4 = Genesis[Genesis.GroupID == 4].index.values.tolist()
    
    dframe['GroupID'] = 0
    dframe.GroupID.loc[dframe.StormID.isin(train_events_g1)] = 1
    dframe.GroupID.loc[dframe.StormID.isin(train_events_g2)] = 2
    dframe.GroupID.loc[dframe.StormID.isin(train_events_g3)] = 3
    dframe.GroupID.loc[dframe.StormID.isin(train_events_g4)] = 4

    one_hot = pd.get_dummies(dframe['GroupID'],prefix='Region')
    train_df = train_df.join(one_hot)
# Creating the model from extrapolated data columns
def model_def(data):
    model = Sequential()
    init = RandomUniform(minval=-0.001, maxval=0.001)
    model.add(LSTM(50,input_shape=(None, data.shape[2]), kernel_initializer=init, return_sequences=True))
    model.add(LSTM(10,input_shape=(None, data.shape[2]), kernel_initializer=init, return_sequences=True))
    model.add(Dense(6, kernel_initializer=init))
    model.add(Dense(3, activation='linear', kernel_initializer=init))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc','mae'])
    return model

def train_on_batch_lstm(data,train_df,NumEpochs):
    model_batch_train = model_def(data)
    for i in range(NumEpochs):
        print ('Running Epoch No: ', i)
        for stormID, data in train_df.groupby('StormID'):
            train = data[['x','y','z','Region_1','Region_2','Region_3','Region_4','S1','S2','S3','S4','S5','S6','S1E', 'S2E', 'S3E', 'S4E', 'S5E', 'S6E','xLag1','yLag1','zLag1']].values
            x_train = np.expand_dims(train[:,:-3], axis=0)
            y_train = np.expand_dims(train[:,-3:], axis=0)
            model_batch_train.train_on_batch(x_train,y_train)
            model_batch_train.reset_states()
    return model_batch_train
  
models = []
for ens in range(ensemble_size):
    lstm_model = train_on_batch_lstm(data,train_df, nb_epoch)
    model_json = lstm_model.to_json()
    with open("gmodel_"+str(ens)+".json", "w") as json_file:
        json_file.write(model_json)
    lstm_model.save_weights("gmodel_"+str(ens)+".h5")
    models.append("gmodel_"+str(ens)+".h5")
    print("Saved model to disk")
# Creating test case for model need to use additional data from previous hurricanes
    
