from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
from datetime import datetime
import pymysql
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.utils import secure_filename
import pandas as pd
import uuid
import json
from collections import Counter, defaultdict
from sqlalchemy import func
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

# Register PyMySQL as the MySQL driver
pymysql.install_as_MySQLdb()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24))

# MySQL Configuration
database_url = os.getenv('DATABASE_URL', 'mysql+pymysql://root:@localhost:3306/crime_system')
if database_url.startswith('mysql://'):
    database_url = database_url.replace('mysql://', 'mysql+pymysql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = True  # This will print SQL queries for debugging

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class CrimeData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    crime_id = db.Column(db.String(64))
    barangay = db.Column(db.String(128))
    type_of_place = db.Column(db.String(128))
    address = db.Column(db.String(255))
    date_reported = db.Column(db.String(32))
    time_reported = db.Column(db.String(32))
    date_committed = db.Column(db.String(32))
    time_committed = db.Column(db.String(32))
    crime_type = db.Column(db.String(128))
    crime_classificaton = db.Column(db.String(128))
    suspect = db.Column(db.String(255))
    victim = db.Column(db.String(255))
    status = db.Column(db.String(64))
    narrative = db.Column(db.Text)
    batch_number = db.Column(db.String(64), index=True)

class Barangay(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), unique=True, nullable=False)
    population = db.Column(db.Integer, default=0)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            password = request.form.get('password')
            remember = True if request.form.get('remember') else False
            
            user = User.query.filter_by(username=username).first()
            
            if not user or not user.check_password(password):
                flash('Please check your login details and try again.')
                return redirect(url_for('login'))
            
            login_user(user, remember=remember)
            return redirect(url_for('index'))
        except SQLAlchemyError as e:
            flash('Database error occurred. Please try again.')
            print(f"Database error: {str(e)}")
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            
            user = User.query.filter_by(username=username).first()
            if user:
                flash('Username already exists')
                return redirect(url_for('register'))
            
            user = User.query.filter_by(email=email).first()
            if user:
                flash('Email already registered')
                return redirect(url_for('register'))
            
            new_user = User(username=username, email=email)
            new_user.set_password(password)
            
            db.session.add(new_user)
            db.session.commit()
            
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        except SQLAlchemyError as e:
            db.session.rollback()
            flash('Database error occurred. Please try again.')
            print(f"Database error: {str(e)}")
            return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/upload_excel', methods=['GET', 'POST'])
@login_required
def upload_excel():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(request.url)
        filename = secure_filename(file.filename)
        if not filename.endswith('.xlsx'):
            flash('Only .xlsx files are allowed.', 'danger')
            return redirect(request.url)
        try:
            df = pd.read_excel(file)
            required_columns = [
                'crime_id', 'barangay', 'type_of_place', 'address', 'date_reported', 'time_reported',
                'date_committed', 'time_committed', 'crime_type', 'crime_classificaton',
                'suspect', 'victim', 'status', 'narrative'
            ]
            if not all(col in df.columns for col in required_columns):
                flash('Excel file is missing required columns.', 'danger')
                return redirect(request.url)
            batch_number = str(uuid.uuid4())[:8]
            for _, row in df.iterrows():
                crime = CrimeData(
                    crime_id=str(row['crime_id']),
                    barangay=row['barangay'],
                    type_of_place=row['type_of_place'],
                    address=row['address'],
                    date_reported=str(row['date_reported']),
                    time_reported=str(row['time_reported']),
                    date_committed=str(row['date_committed']),
                    time_committed=str(row['time_committed']),
                    crime_type=row['crime_type'],
                    crime_classificaton=row['crime_classificaton'],
                    suspect=row['suspect'],
                    victim=row['victim'],
                    status=row['status'],
                    narrative=row['narrative'],
                    batch_number=batch_number
                )
                db.session.add(crime)
            db.session.commit()
            flash(f'Successfully uploaded! Batch number: {batch_number}', 'success')
            return redirect(url_for('upload_excel'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error processing file: {e}', 'danger')
            return redirect(request.url)
    return render_template('upload_excel.html')

@app.route('/view_crime_data')
@login_required
def view_crime_data():
    # Get filter, search, and sort parameters from query string
    search = request.args.get('search', '', type=str)
    batch_number = request.args.get('batch_number', '', type=str)
    barangay = request.args.get('barangay', '', type=str)
    crime_type = request.args.get('crime_type', '', type=str)
    status = request.args.get('status', '', type=str)
    sort_by = request.args.get('sort_by', 'id', type=str)
    sort_dir = request.args.get('sort_dir', 'desc', type=str)
    page = request.args.get('page', 1, type=int)
    per_page = 7

    # Build query
    query = CrimeData.query
    if search:
        like = f"%{search}%"
        query = query.filter(
            (CrimeData.crime_id.like(like)) |
            (CrimeData.barangay.like(like)) |
            (CrimeData.type_of_place.like(like)) |
            (CrimeData.address.like(like)) |
            (CrimeData.date_reported.like(like)) |
            (CrimeData.time_reported.like(like)) |
            (CrimeData.date_committed.like(like)) |
            (CrimeData.time_committed.like(like)) |
            (CrimeData.crime_type.like(like)) |
            (CrimeData.crime_classificaton.like(like)) |
            (CrimeData.suspect.like(like)) |
            (CrimeData.victim.like(like)) |
            (CrimeData.status.like(like)) |
            (CrimeData.narrative.like(like)) |
            (CrimeData.batch_number.like(like))
        )
    if batch_number:
        query = query.filter(CrimeData.batch_number == batch_number)
    if barangay:
        query = query.filter(CrimeData.barangay == barangay)
    if crime_type:
        query = query.filter(CrimeData.crime_type == crime_type)
    if status:
        query = query.filter(CrimeData.status == status)

    # Sorting
    sort_column = getattr(CrimeData, sort_by, CrimeData.id)
    if sort_dir == 'desc':
        sort_column = sort_column.desc()
    else:
        sort_column = sort_column.asc()
    query = query.order_by(sort_column)

    # Fetch all records (for now, can add pagination later)
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    records = pagination.items

    # For filter dropdowns
    batch_numbers = [r[0] for r in db.session.query(CrimeData.batch_number).distinct().all() if r[0]]
    barangays = [r[0] for r in db.session.query(CrimeData.barangay).distinct().all() if r[0]]
    crime_types = [r[0] for r in db.session.query(CrimeData.crime_type).distinct().all() if r[0]]
    statuses = [r[0] for r in db.session.query(CrimeData.status).distinct().all() if r[0]]

    return render_template(
        'view_crime_data.html',
        records=records,
        batch_numbers=batch_numbers,
        barangays=barangays,
        crime_types=crime_types,
        statuses=statuses,
        search=search,
        selected_batch=batch_number,
        selected_barangay=barangay,
        selected_crime_type=crime_type,
        selected_status=status,
        sort_by=sort_by,
        sort_dir=sort_dir,
        pagination=pagination
    )

@app.route('/view_barangay', methods=['GET', 'POST'])
@login_required
def view_barangay():
    if request.method == 'POST':
        barangay_id = request.form.get('barangay_id')
        new_population = request.form.get('population')
        try:
            barangay = Barangay.query.get(barangay_id)
            if barangay and new_population.isdigit():
                barangay.population = int(new_population)
                db.session.commit()
                flash(f'Population for {barangay.name} updated!', 'success')
            else:
                flash('Invalid input.', 'danger')
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating population: {e}', 'danger')
        return redirect(url_for('view_barangay'))

    # Search, filter, and sort
    search = request.args.get('search', '', type=str)
    pop_min = request.args.get('pop_min', '', type=str)
    pop_max = request.args.get('pop_max', '', type=str)
    sort_by = request.args.get('sort_by', 'name', type=str)
    sort_dir = request.args.get('sort_dir', 'asc', type=str)

    query = Barangay.query
    if search:
        query = query.filter(Barangay.name.ilike(f"%{search}%"))
    if pop_min.isdigit():
        query = query.filter(Barangay.population >= int(pop_min))
    if pop_max.isdigit():
        query = query.filter(Barangay.population <= int(pop_max))
    sort_column = getattr(Barangay, sort_by, Barangay.name)
    if sort_dir == 'desc':
        sort_column = sort_column.desc()
    else:
        sort_column = sort_column.asc()
    query = query.order_by(sort_column)
    barangays = query.all()
    return render_template('view_barangay.html', barangays=barangays, search=search, pop_min=pop_min, pop_max=pop_max, sort_by=sort_by, sort_dir=sort_dir)

@app.route('/dashboard')
@login_required
def dashboard():
    # Get unique values for filters
    years = sorted([r[0] for r in db.session.query(func.substr(CrimeData.date_reported, 1, 4)).distinct().all() if r[0]])
    barangays = sorted([r[0] for r in db.session.query(CrimeData.barangay).distinct().all() if r[0]])
    crime_types = sorted([r[0] for r in db.session.query(CrimeData.crime_type).distinct().all() if r[0]])
    statuses = sorted([r[0] for r in db.session.query(CrimeData.status).distinct().all() if r[0]])
    batch_numbers = sorted([r[0] for r in db.session.query(CrimeData.batch_number).distinct().all() if r[0]])

    return render_template('dashboard.html',
        years=years,
        barangays=barangays,
        crime_types=crime_types,
        statuses=statuses,
        batch_numbers=batch_numbers
    )

@app.route('/api/dashboard-data')
@login_required
def dashboard_data():
    try:
        batch_number = request.args.get('batch_number', '')
        year = request.args.get('year', '')
        barangay = request.args.get('barangay', '')
        crime_type = request.args.get('crime_type', '')
        status = request.args.get('status', '')

        print("Starting dashboard data processing...")
        print(f"Filters: batch={batch_number}, year={year}, barangay={barangay}, crime_type={crime_type}, status={status}")

        # Base query
        query = CrimeData.query

        # Apply filters
        if batch_number:
            query = query.filter(CrimeData.batch_number == batch_number)
        if year:
            query = query.filter(CrimeData.date_reported.like(f"{year}%"))
        if barangay:
            query = query.filter(CrimeData.barangay == barangay)
        if crime_type:
            query = query.filter(CrimeData.crime_type == crime_type)
        if status:
            query = query.filter(CrimeData.status == status)

        # Get filtered data
        filtered_data = query.all()
        print(f"Filtered data count: {len(filtered_data)}")

        try:
            # 1. Crimes Over Time (Monthly)
            monthly_data = db.session.query(
                func.substr(CrimeData.date_reported, 1, 7),
                func.count()
            ).filter(CrimeData.id.in_([d.id for d in filtered_data])).group_by(
                func.substr(CrimeData.date_reported, 1, 7)
            ).all()
            print("Monthly data processed")

            # 2. Crime Type Distribution
            crime_type_data = db.session.query(
                CrimeData.crime_type,
                func.count()
            ).filter(CrimeData.id.in_([d.id for d in filtered_data])).group_by(
                CrimeData.crime_type
            ).all()
            print("Crime type data processed")

            # 3. Crime Density by Barangay
            density_data = db.session.query(
                CrimeData.barangay,
                func.count()
            ).filter(CrimeData.id.in_([d.id for d in filtered_data])).group_by(
                CrimeData.barangay
            ).all()
            print("Density data processed")

            # 4. Crime Rate by Barangay
            barangay_pop = {b.name: b.population for b in Barangay.query.all()}
            crime_counts = dict(density_data)
            crime_rate_data = []
            for barangay, count in crime_counts.items():
                pop = barangay_pop.get(barangay, 0)
                rate = (count / pop) * 100000 if pop > 0 else 0
                crime_rate_data.append((barangay, round(rate, 2)))
            print("Crime rate data processed")

            # 5. Crime Type by Barangay
            crime_type_by_barangay = db.session.query(
                CrimeData.barangay,
                CrimeData.crime_type,
                func.count()
            ).filter(CrimeData.id.in_([d.id for d in filtered_data])).group_by(
                CrimeData.barangay,
                CrimeData.crime_type
            ).all()
            print("Crime type by barangay data processed")

            # 6. Time of Day Distribution
            hour_data = [0] * 24
            for record in filtered_data:
                try:
                    if record.time_committed:
                        hour = int(str(record.time_committed).split(':')[0])
                        if 0 <= hour < 24:
                            hour_data[hour] += 1
                except (ValueError, AttributeError) as e:
                    print(f"Error processing time for record {record.id}: {str(e)}")
                    continue
            print("Time of day data processed")

            # 7. Resolution Status
            status_data = db.session.query(
                CrimeData.status,
                func.count()
            ).filter(CrimeData.id.in_([d.id for d in filtered_data])).group_by(
                CrimeData.status
            ).all()
            print("Resolution status data processed")

            # 8. Place Type Distribution
            place_type_data = db.session.query(
                CrimeData.type_of_place,
                func.count()
            ).filter(CrimeData.id.in_([d.id for d in filtered_data])).group_by(
                CrimeData.type_of_place
            ).all()
            print("Place type data processed")

            # Prepare response data
            response = {
                'crimes_over_time': {
                    'labels': [d[0] for d in monthly_data] if monthly_data else [],
                    'values': [d[1] for d in monthly_data] if monthly_data else [],
                    'insight': f"Peak month: {max(monthly_data, key=lambda x: x[1])[0] if monthly_data else 'N/A'}"
                },
                'crime_type': {
                    'labels': [d[0] for d in crime_type_data] if crime_type_data else [],
                    'values': [d[1] for d in crime_type_data] if crime_type_data else [],
                    'insight': f"Most common: {max(crime_type_data, key=lambda x: x[1])[0] if crime_type_data else 'N/A'}"
                },
                'crime_density': {
                    'labels': [d[0] for d in density_data] if density_data else [],
                    'values': [d[1] for d in density_data] if density_data else [],
                    'insight': f"Highest density: {max(density_data, key=lambda x: x[1])[0] if density_data else 'N/A'}"
                },
                'crime_rate': {
                    'labels': [d[0] for d in crime_rate_data] if crime_rate_data else [],
                    'values': [d[1] for d in crime_rate_data] if crime_rate_data else [],
                    'insight': f"Highest rate: {max(crime_rate_data, key=lambda x: x[1])[0] if crime_rate_data else 'N/A'}"
                },
                'crime_type_by_barangay': {
                    'labels': sorted(set(d[0] for d in crime_type_by_barangay)) if crime_type_by_barangay else [],
                    'datasets': [{
                        'label': crime_type,
                        'data': [next((d[2] for d in crime_type_by_barangay if d[0] == barangay and d[1] == crime_type), 0)
                                for barangay in sorted(set(d[0] for d in crime_type_by_barangay))] if crime_type_by_barangay else [],
                        'backgroundColor': f'rgba({i * 50}, {255 - i * 50}, {i * 30}, 0.7)'
                    } for i, crime_type in enumerate(sorted(set(d[1] for d in crime_type_by_barangay))) if crime_type_by_barangay],
                    'insight': "Distribution of crime types across barangays"
                },
                'time_of_day': {
                    'labels': [f"{h:02d}:00" for h in range(24)],
                    'values': hour_data,
                    'insight': f"Peak hour: {hour_data.index(max(hour_data)):02d}:00" if max(hour_data) > 0 else "No data"
                },
                'resolution_status': {
                    'labels': [d[0] for d in status_data] if status_data else [],
                    'values': [d[1] for d in status_data] if status_data else [],
                    'insight': f"Most common: {max(status_data, key=lambda x: x[1])[0] if status_data else 'N/A'}"
                },
                'place_type': {
                    'labels': [d[0] for d in place_type_data] if place_type_data else [],
                    'values': [d[1] for d in place_type_data] if place_type_data else [],
                    'insight': f"Most common: {max(place_type_data, key=lambda x: x[1])[0] if place_type_data else 'N/A'}"
                }
            }

            print("Response data prepared successfully")
            return jsonify(response)

        except Exception as e:
            print(f"Error processing dashboard data: {str(e)}")
            return jsonify({
                'error': str(e),
                'message': 'An error occurred while processing the dashboard data'
            }), 500

    except Exception as e:
        print(f"Error in dashboard_data endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'An error occurred in the dashboard data endpoint'
        }), 500

@app.route('/api/clustering-analysis')
@login_required
def clustering_analysis():
    try:
        # Get filter parameters
        batch_number = request.args.get('batch_number', '')
        year = request.args.get('year', '')
        
        # Base query
        query = CrimeData.query
        
        # Apply filters
        if batch_number:
            query = query.filter(CrimeData.batch_number == batch_number)
        if year:
            query = query.filter(CrimeData.date_reported.like(f"{year}%"))
        
        # Get all crime data
        crime_data = query.all()
        
        def find_optimal_clusters(X, max_clusters=10):
            """Find optimal number of clusters using multiple metrics"""
            if len(X) < 2:
                return 2, {}
            
            metrics = {
                'silhouette': [],
                'calinski_harabasz': [],
                'davies_bouldin': []
            }
            
            max_clusters = min(max_clusters, len(X) - 1)
            n_clusters_range = range(2, max_clusters + 1)
            
            for n_clusters in n_clusters_range:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X)
                
                # Calculate metrics
                metrics['silhouette'].append(silhouette_score(X, cluster_labels))
                metrics['calinski_harabasz'].append(calinski_harabasz_score(X, cluster_labels))
                metrics['davies_bouldin'].append(davies_bouldin_score(X, cluster_labels))
            
            # Find optimal clusters based on silhouette score
            optimal_clusters = n_clusters_range[np.argmax(metrics['silhouette'])]
            
            return optimal_clusters, metrics

        def dbscan_clustering(X, eps=0.5, min_samples=2):
            if len(X) < min_samples:
                return None, None, None
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            # Silhouette score only if more than 1 cluster and not all noise
            if n_clusters > 1 and n_noise < len(X):
                sil_score = silhouette_score(X, labels)
            else:
                sil_score = None
            return labels, n_clusters, {'n_clusters': n_clusters, 'n_noise': n_noise, 'silhouette': sil_score}

        def perform_crime_rate_clustering():
            # Get crime counts and population for each barangay
            barangay_pop = {b.name: b.population for b in Barangay.query.all()}
            crime_counts = defaultdict(int)
            
            for crime in crime_data:
                crime_counts[crime.barangay] += 1
            
            # Calculate crime rates and additional features
            features = []
            barangay_names = []
            for barangay, count in crime_counts.items():
                pop = barangay_pop.get(barangay, 0)
                if pop > 0:
                    rate = (count / pop) * 100000  # Crimes per 100,000 people
                    # Add multiple features for better clustering
                    features.append([rate, count, pop])
                    barangay_names.append(barangay)
            
            if not features:
                return None, None, None, None
            
            # Scale features using RobustScaler (more robust to outliers)
            X = np.array(features)
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Find optimal number of clusters
            optimal_clusters, metrics = find_optimal_clusters(X_scaled)
            
            # Perform final clustering
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Calculate cluster statistics
            cluster_stats = []
            for i in range(optimal_clusters):
                cluster_data = X[cluster_labels == i]
                cluster_stats.append({
                    'mean': np.mean(cluster_data, axis=0),
                    'std': np.std(cluster_data, axis=0),
                    'min': np.min(cluster_data, axis=0),
                    'max': np.max(cluster_data, axis=0),
                    'size': len(cluster_data)
                })
            
            # Assign risk levels based on statistical analysis
            risk_labels = {}
            sorted_clusters = sorted(range(len(cluster_stats)), 
                                  key=lambda i: cluster_stats[i]['mean'][0])  # Sort by crime rate
            
            risk_levels = ["Low Risk Area", "Medium-Low Risk Area", 
                         "Medium-High Risk Area", "High Risk Area"]
            
            for i, cluster_idx in enumerate(sorted_clusters):
                risk_labels[cluster_idx] = risk_levels[min(i, len(risk_levels)-1)]
            
            # Prepare results with confidence scores and additional metrics
            results = []
            for i, (barangay, feature) in enumerate(zip(barangay_names, features)):
                cluster = int(cluster_labels[i])
                cluster_stat = cluster_stats[cluster]
                
                # Calculate confidence score based on multiple factors
                center = cluster_stat['mean']
                std = cluster_stat['std']
                distance = np.linalg.norm(feature - center)
                avg_std = np.mean(std)
                confidence = max(0, 100 - (distance / avg_std * 50))
                
                # Calculate additional metrics
                crime_rate = feature[0]
                crime_count = feature[1]
                population = feature[2]
                
                results.append({
                    'barangay': barangay,
                    'crime_rate': round(crime_rate, 2),
                    'crime_count': int(crime_count),
                    'population': int(population),
                    'cluster': cluster,
                    'cluster_label': risk_labels[cluster],
                    'confidence': round(confidence, 2),
                    'metrics': {
                        'distance_to_center': round(distance, 2),
                        'cluster_size': cluster_stat['size']
                    }
                })
            
            return results, metrics, optimal_clusters, X_scaled

        def perform_hotspot_clustering():
            # Count crimes per barangay and add temporal features
            crime_counts = defaultdict(lambda: {'count': 0, 'times': []})
            for crime in crime_data:
                crime_counts[crime.barangay]['count'] += 1
                if crime.time_committed:
                    try:
                        hour = int(str(crime.time_committed).split(':')[0])
                        crime_counts[crime.barangay]['times'].append(hour)
                    except (ValueError, AttributeError):
                        continue
            
            # Prepare features for clustering
            features = []
            barangay_names = []
            for barangay, data in crime_counts.items():
                count = data['count']
                times = data['times']
                if times:
                    # Calculate temporal features
                    peak_hour = max(set(times), key=times.count) if times else 0
                    night_crimes = sum(1 for t in times if 22 <= t <= 4)
                    day_crimes = sum(1 for t in times if 5 <= t <= 17)
                    evening_crimes = sum(1 for t in times if 18 <= t <= 21)
                    
                    features.append([count, peak_hour, night_crimes, day_crimes, evening_crimes])
                    barangay_names.append(barangay)
            
            if not features:
                return None, None, None, None
            
            # Scale features
            X = np.array(features)
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Find optimal number of clusters
            optimal_clusters, metrics = find_optimal_clusters(X_scaled)
            
            # Perform final clustering
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Calculate cluster statistics
            cluster_stats = []
            for i in range(optimal_clusters):
                cluster_data = X[cluster_labels == i]
                cluster_stats.append({
                    'mean': np.mean(cluster_data, axis=0),
                    'std': np.std(cluster_data, axis=0),
                    'size': len(cluster_data)
                })
            
            # Assign density levels based on statistical analysis
            density_labels = {}
            sorted_clusters = sorted(range(len(cluster_stats)), 
                                  key=lambda i: cluster_stats[i]['mean'][0])  # Sort by crime count
            
            density_levels = ["Low Crime Density", "Medium-Low Crime Density", 
                            "Medium-High Crime Density", "High Crime Density"]
            
            for i, cluster_idx in enumerate(sorted_clusters):
                density_labels[cluster_idx] = density_levels[min(i, len(density_levels)-1)]
            
            # Prepare results with confidence scores and temporal patterns
            results = []
            for i, (barangay, feature) in enumerate(zip(barangay_names, features)):
                cluster = int(cluster_labels[i])
                cluster_stat = cluster_stats[cluster]
                
                # Calculate confidence score
                center = cluster_stat['mean']
                std = cluster_stat['std']
                distance = np.linalg.norm(feature - center)
                avg_std = np.mean(std)
                confidence = max(0, 100 - (distance / avg_std * 50))
                
                # Determine temporal pattern
                count, peak_hour, night, day, evening = feature
                total = night + day + evening
                if total > 0:
                    night_pct = (night / total) * 100
                    day_pct = (day / total) * 100
                    evening_pct = (evening / total) * 100
                    
                    if night_pct > 50:
                        temporal_pattern = "Night-time Crime Pattern"
                    elif day_pct > 50:
                        temporal_pattern = "Day-time Crime Pattern"
                    elif evening_pct > 50:
                        temporal_pattern = "Evening Crime Pattern"
                    else:
                        temporal_pattern = "Mixed Time Pattern"
                else:
                    temporal_pattern = "No Temporal Pattern"
                
                results.append({
                    'barangay': barangay,
                    'crime_count': int(feature[0]),
                    'peak_hour': int(feature[1]),
                    'temporal_pattern': temporal_pattern,
                    'time_distribution': {
                        'night': int(feature[2]),
                        'day': int(feature[3]),
                        'evening': int(feature[4])
                    },
                    'cluster': cluster,
                    'cluster_label': density_labels[cluster],
                    'confidence': round(confidence, 2),
                    'metrics': {
                        'distance_to_center': round(distance, 2),
                        'cluster_size': cluster_stat['size']
                    }
                })
            
            return results, metrics, optimal_clusters, X_scaled

        def perform_crime_type_clustering():
            # Count crimes by type for each barangay
            crime_type_counts = defaultdict(lambda: defaultdict(int))
            for crime in crime_data:
                crime_type_counts[crime.barangay][crime.crime_type] += 1
            
            # Get all unique crime types
            all_crime_types = sorted(set(crime.crime_type for crime in crime_data))
            
            # Prepare features for clustering
            features = []
            barangay_names = []
            for barangay, type_counts in crime_type_counts.items():
                row = [type_counts.get(crime_type, 0) for crime_type in all_crime_types]
                features.append(row)
                barangay_names.append(barangay)
            
            if not features:
                return None, None, None, None
            
            # Scale features
            X = np.array(features)
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply PCA for dimensionality reduction if needed
            if X_scaled.shape[1] > 10:
                pca = PCA(n_components=10)
                X_scaled = pca.fit_transform(X_scaled)
            
            # Find optimal number of clusters
            optimal_clusters, metrics = find_optimal_clusters(X_scaled)
            
            # Perform final clustering
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Calculate cluster statistics
            cluster_stats = []
            for i in range(optimal_clusters):
                cluster_data = X[cluster_labels == i]
                cluster_stats.append({
                    'mean': np.mean(cluster_data, axis=0),
                    'std': np.std(cluster_data, axis=0),
                    'size': len(cluster_data)
                })
            
            # Assign pattern labels based on dominant crime types
            pattern_labels = {}
            for i in range(optimal_clusters):
                # Find top 3 crime types for this cluster
                mean_counts = cluster_stats[i]['mean']
                top_indices = np.argsort(mean_counts)[-3:][::-1]
                top_crimes = [all_crime_types[idx] for idx in top_indices]
                
                # Calculate crime type distribution
                total_crimes = np.sum(mean_counts)
                if total_crimes > 0:
                    top_percentages = [(mean_counts[idx] / total_crimes) * 100 for idx in top_indices]
                    pattern_labels[i] = {
                        'label': f"Pattern: {', '.join(top_crimes)}",
                        'distribution': dict(zip(top_crimes, [round(p, 1) for p in top_percentages]))
                    }
                else:
                    pattern_labels[i] = {
                        'label': "No Dominant Pattern",
                        'distribution': {}
                    }
            
            # Prepare results with confidence scores and detailed patterns
            results = []
            for i, (barangay, counts) in enumerate(zip(barangay_names, X)):
                cluster = int(cluster_labels[i])
                cluster_stat = cluster_stats[cluster]
                
                # Calculate confidence score
                center = cluster_stat['mean']
                std = cluster_stat['std']
                distance = np.linalg.norm(counts - center)
                avg_std = np.mean(std)
                confidence = max(0, 100 - (distance / avg_std * 50))
                
                # Get top crime types for this barangay
                total_crimes = np.sum(counts)
                if total_crimes > 0:
                    top_indices = np.argsort(counts)[-3:][::-1]
                    top_crimes = [all_crime_types[idx] for idx in top_indices]
                    top_percentages = [(counts[idx] / total_crimes) * 100 for idx in top_indices]
                    crime_distribution = dict(zip(top_crimes, [round(p, 1) for p in top_percentages]))
                else:
                    crime_distribution = {}
                
                results.append({
                    'barangay': barangay,
                    'crime_types': {crime_type: int(counts[j]) for j, crime_type in enumerate(all_crime_types)},
                    'crime_distribution': crime_distribution,
                    'cluster': cluster,
                    'cluster_label': pattern_labels[cluster]['label'],
                    'cluster_pattern': pattern_labels[cluster]['distribution'],
                    'confidence': round(confidence, 2),
                    'metrics': {
                        'distance_to_center': round(distance, 2),
                        'cluster_size': cluster_stat['size']
                    }
                })
            
            return results, metrics, optimal_clusters, X_scaled

        # Perform all clustering analyses (KMeans)
        crime_rate_clusters, crime_rate_metrics, crime_rate_optimal, crime_rate_X = perform_crime_rate_clustering()
        hotspot_clusters, hotspot_metrics, hotspot_optimal, hotspot_X = perform_hotspot_clustering()
        crime_type_clusters, crime_type_metrics, crime_type_optimal, crime_type_X = perform_crime_type_clustering()

        # Perform DBSCAN for each analysis
        dbscan_results = {}
        # Crime Rate DBSCAN
        if crime_rate_X is not None:
            labels, n_clusters, dbscan_metrics = dbscan_clustering(crime_rate_X)
            dbscan_results['crime_rate'] = {
                'labels': labels.tolist() if labels is not None else [],
                'metrics': dbscan_metrics
            }
        else:
            dbscan_results['crime_rate'] = {'labels': [], 'metrics': {}}
        # Hotspot DBSCAN
        if hotspot_X is not None:
            labels, n_clusters, dbscan_metrics = dbscan_clustering(hotspot_X)
            dbscan_results['hotspot'] = {
                'labels': labels.tolist() if labels is not None else [],
                'metrics': dbscan_metrics
            }
        else:
            dbscan_results['hotspot'] = {'labels': [], 'metrics': {}}
        # Crime Type DBSCAN
        if crime_type_X is not None:
            labels, n_clusters, dbscan_metrics = dbscan_clustering(crime_type_X)
            dbscan_results['crime_type'] = {
                'labels': labels.tolist() if labels is not None else [],
                'metrics': dbscan_metrics
            }
        else:
            dbscan_results['crime_type'] = {'labels': [], 'metrics': {}}

        return jsonify({
            'crime_rate_clusters': crime_rate_clusters,
            'crime_rate_metrics': crime_rate_metrics,
            'crime_rate_optimal_clusters': crime_rate_optimal,
            'hotspot_clusters': hotspot_clusters,
            'hotspot_metrics': hotspot_metrics,
            'hotspot_optimal_clusters': hotspot_optimal,
            'crime_type_clusters': crime_type_clusters,
            'crime_type_metrics': crime_type_metrics,
            'crime_type_optimal_clusters': crime_type_optimal,
            'dbscan': dbscan_results
        })

    except Exception as e:
        print(f"Error in clustering analysis: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'An error occurred during clustering analysis'
        }), 500

if __name__ == '__main__':
    try:
        # Test database connection
        with app.app_context():
            try:
                db.engine.connect()
                print("Successfully connected to MySQL database!")
            except Exception as e:
                print(f"Error connecting to MySQL: {str(e)}")
                print("\nPlease check the following:")
                print("1. Is XAMPP running?")
                print("2. Is MySQL service started in XAMPP?")
                print("3. Can you access phpMyAdmin at http://localhost/phpmyadmin?")
                print("4. Is the 'crime_system' database created?")
                print("\nTo create the database:")
                print("1. Open http://localhost/phpmyadmin")
                print("2. Click 'New' on the left sidebar")
                print("3. Enter 'crime_system' as the database name")
                print("4. Click 'Create'")
                raise e

            db.create_all()
            print("Database tables created successfully!")
        app.run(debug=True)
    except Exception as e:
        print(f"Error starting the application: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Open XAMPP Control Panel")
        print("2. Stop MySQL if it's running")
        print("3. Start MySQL again")
        print("4. Wait a few seconds for MySQL to start completely")
        print("5. Try running the application again")
        print("\nPlease make sure all required packages are installed:")
        print("pip install -r requirements.txt") 