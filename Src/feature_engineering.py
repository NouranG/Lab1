def create_features(df):
    df = df.copy()

    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Ticket group size
    df['TicketGroupSize'] = df.groupby('Ticket')['Ticket'].transform('count')

    # Title extraction
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

    # Ticket prefix
    df['TicketPrefix'] = df['Ticket'].str.extract(r'([A-Za-z]+)')

    return df