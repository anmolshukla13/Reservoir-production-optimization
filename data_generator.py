"""
Synthetic Reservoir Production Data Generator
==============================================
Generates realistic oil & gas production data based on reservoir engineering principles.

This simulates:
- Multiple wells with different characteristics
- Production decline curves (Arps decline)
- Reservoir pressure depletion
- Water cut evolution
- Gas-oil ratio changes
- Well interference effects
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ReservoirDataGenerator:
    """Generate synthetic reservoir production data"""
    
    def __init__(self, n_wells=50, n_days=1095, random_state=42):
        """
        Initialize data generator
        
        Args:
            n_wells: Number of wells to simulate
            n_days: Number of days of production data (default: 3 years)
            random_state: Random seed for reproducibility
        """
        self.n_wells = n_wells
        self.n_days = n_days
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_well_properties(self):
        """Generate static well and reservoir properties"""
        
        wells = []
        for i in range(self.n_wells):
            well = {
                'well_id': f'WELL_{i+1:03d}',
                'well_type': np.random.choice(['Producer', 'Injector'], p=[0.85, 0.15]),
                'completion_date': datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 365)),
                
                # Reservoir properties
                'porosity': np.random.normal(0.22, 0.05),  # 22% average
                'permeability': np.random.lognormal(2.5, 1.2),  # mD (log-normal distribution)
                'net_pay': np.random.normal(25, 8),  # meters
                'initial_pressure': np.random.normal(3500, 300),  # psi
                'reservoir_temperature': np.random.normal(180, 15),  # °F
                
                # Well properties
                'measured_depth': np.random.normal(8500, 1200),  # feet
                'true_vertical_depth': np.random.normal(7800, 1000),  # feet
                'wellbore_radius': 0.328,  # feet (standard 7-7/8" hole)
                'skin_factor': np.random.normal(2, 3),  # dimensionless
                
                # Completion properties
                'tubing_diameter': np.random.choice([2.875, 3.5, 4.5]),  # inches
                'choke_size': np.random.choice([16, 20, 24, 32, 40, 48, 64]),  # 64ths of inch
                
                # PVT properties
                'oil_api': np.random.normal(35, 5),  # API gravity
                'gas_gravity': np.random.normal(0.65, 0.08),  # specific gravity
                'water_salinity': np.random.normal(35000, 5000),  # ppm
                
                # Economic zone
                'field_zone': np.random.choice(['North', 'South', 'East', 'West']),
                
                # Initial rates (for Arps decline)
                'initial_oil_rate': np.random.lognormal(5.5, 0.8) * 100,  # bbl/day
                'initial_gas_rate': np.random.lognormal(5.5, 0.8) * 1000,  # Mcf/day
                'initial_water_rate': np.random.lognormal(3, 0.5) * 10,  # bbl/day
                
                # Decline parameters
                'decline_rate': np.random.uniform(0.15, 0.45),  # nominal decline (1/year)
                'b_factor': np.random.uniform(0.2, 1.5),  # Arps b-factor
                'eur': np.random.lognormal(12, 0.8),  # Estimated Ultimate Recovery (MMbbl)
                
                # Coordinates
                'x_coordinate': np.random.uniform(0, 10000),  # feet
                'y_coordinate': np.random.uniform(0, 10000),  # feet
            }
            
            # Ensure physical constraints
            well['porosity'] = np.clip(well['porosity'], 0.05, 0.40)
            well['permeability'] = np.clip(well['permeability'], 0.1, 5000)
            well['net_pay'] = np.clip(well['net_pay'], 5, 100)
            well['skin_factor'] = np.clip(well['skin_factor'], -5, 15)
            well['oil_api'] = np.clip(well['oil_api'], 15, 50)
            
            wells.append(well)
        
        return pd.DataFrame(wells)
    
    def arps_decline(self, qi, Di, b, t):
        """
        Arps hyperbolic decline equation
        
        Args:
            qi: Initial production rate
            Di: Initial decline rate (nominal)
            b: Hyperbolic exponent
            t: Time in days
        
        Returns:
            Production rate at time t
        """
        t_years = t / 365.25
        if b == 0:  # Exponential decline
            return qi * np.exp(-Di * t_years)
        else:  # Hyperbolic decline
            return qi / ((1 + b * Di * t_years) ** (1/b))
    
    def generate_production_data(self, well_properties):
        """Generate daily production data for all wells"""
        
        production_data = []
        
        for _, well in well_properties.iterrows():
            if well['well_type'] == 'Injector':
                continue  # Skip injectors for now
            
            completion_date = well['completion_date']
            days_since_completion = 0
            
            for day in range(self.n_days):
                current_date = datetime(2020, 1, 1) + timedelta(days=day)
                
                if current_date < completion_date:
                    continue  # Well not yet completed
                
                days_since_completion = (current_date - completion_date).days
                
                # Apply Arps decline
                oil_rate = self.arps_decline(
                    well['initial_oil_rate'],
                    well['decline_rate'],
                    well['b_factor'],
                    days_since_completion
                )
                
                gas_rate = self.arps_decline(
                    well['initial_gas_rate'],
                    well['decline_rate'] * 0.8,  # Gas declines slower
                    well['b_factor'],
                    days_since_completion
                )
                
                # Water cut increases over time (waterflood)
                initial_wc = well['initial_water_rate'] / (well['initial_oil_rate'] + well['initial_water_rate'])
                time_factor = 1 - np.exp(-days_since_completion / 730)  # 2-year time constant
                water_cut = initial_wc + (0.85 - initial_wc) * time_factor
                water_cut = np.clip(water_cut, 0, 0.98)
                
                total_liquid = oil_rate / (1 - water_cut)
                water_rate = total_liquid * water_cut
                oil_rate = total_liquid * (1 - water_cut)
                
                # Gas-Oil Ratio evolves
                gor_initial = well['initial_gas_rate'] / well['initial_oil_rate']
                gor = gor_initial * (1 + 0.3 * time_factor)  # GOR increases
                gas_rate = oil_rate * gor
                
                # Reservoir pressure depletion
                pressure_decline = well['initial_pressure'] * 0.0003 * days_since_completion
                reservoir_pressure = well['initial_pressure'] - pressure_decline
                reservoir_pressure = max(reservoir_pressure, 500)  # Minimum abandonment pressure
                
                # Wellhead pressure (function of choke size and rates)
                wellhead_pressure = reservoir_pressure * 0.2 + (64 - well['choke_size']) * 5
                
                # Add noise and operational issues
                oil_rate *= np.random.normal(1.0, 0.05)  # 5% daily variation
                gas_rate *= np.random.normal(1.0, 0.05)
                water_rate *= np.random.normal(1.0, 0.08)
                
                # Random shutdowns (1% chance per day)
                if np.random.random() < 0.01:
                    oil_rate *= 0.1
                    gas_rate *= 0.1
                    water_rate *= 0.1
                
                # Ensure non-negative
                oil_rate = max(oil_rate, 0)
                gas_rate = max(gas_rate, 0)
                water_rate = max(water_rate, 0)
                
                production_data.append({
                    'well_id': well['well_id'],
                    'date': current_date,
                    'days_on_production': days_since_completion,
                    'oil_rate': oil_rate,  # bbl/day
                    'gas_rate': gas_rate,  # Mcf/day
                    'water_rate': water_rate,  # bbl/day
                    'reservoir_pressure': reservoir_pressure,  # psi
                    'wellhead_pressure': wellhead_pressure,  # psi
                    'choke_size': well['choke_size'],  # 64ths
                    'water_cut': water_cut * 100,  # percent
                    'gor': gor,  # scf/bbl
                    'status': 'Producing' if oil_rate > 5 else 'Shut-in'
                })
        
        return pd.DataFrame(production_data)
    
    def add_operational_data(self, production_df):
        """Add operational variables"""
        
        # Calculate cumulative production
        production_df = production_df.sort_values(['well_id', 'date'])
        production_df['cumulative_oil'] = production_df.groupby('well_id')['oil_rate'].cumsum()
        production_df['cumulative_gas'] = production_df.groupby('well_id')['gas_rate'].cumsum()
        production_df['cumulative_water'] = production_df.groupby('well_id')['water_rate'].cumsum()
        
        # Total liquid rate
        production_df['liquid_rate'] = production_df['oil_rate'] + production_df['water_rate']
        
        # Pressure drawdown
        production_df['drawdown'] = production_df['reservoir_pressure'] - production_df['wellhead_pressure']
        
        # Productivity index (simplified)
        production_df['productivity_index'] = production_df['liquid_rate'] / (production_df['drawdown'] + 1)
        
        return production_df
    
    def generate_complete_dataset(self):
        """Generate complete dataset with wells and production"""
        
        print("Generating well properties...")
        well_properties = self.generate_well_properties()
        
        print("Generating production data...")
        production_data = self.generate_production_data(well_properties)
        
        print("Adding operational features...")
        production_data = self.add_operational_data(production_data)
        
        # Merge with well properties
        full_data = production_data.merge(well_properties, on='well_id', how='left')
        
        print(f"\nDataset generated successfully!")
        print(f"Wells: {self.n_wells}")
        print(f"Total records: {len(full_data):,}")
        print(f"Date range: {full_data['date'].min()} to {full_data['date'].max()}")
        
        return well_properties, production_data, full_data


def main():
    """Generate and save synthetic datasets"""
    
    # Generate data
    generator = ReservoirDataGenerator(n_wells=50, n_days=1095, random_state=42)
    well_props, production, full_data = generator.generate_complete_dataset()
    
    # Create data directory
    import os
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/synthetic', exist_ok=True)
    
    # Save datasets
    print("\nSaving datasets...")
    well_props.to_csv('data/synthetic/well_properties.csv', index=False)
    production.to_csv('data/synthetic/production_data.csv', index=False)
    full_data.to_csv('data/synthetic/full_dataset.csv', index=False)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    print("\nWell Properties:")
    print(well_props.describe())
    
    print("\nProduction Data Summary:")
    print(production.describe())
    
    print("\nProduction by Well Type:")
    print(well_props['well_type'].value_counts())
    
    print("\nAverage Production Rates:")
    print(f"Oil: {production['oil_rate'].mean():.2f} bbl/day")
    print(f"Gas: {production['gas_rate'].mean():.2f} Mcf/day")
    print(f"Water: {production['water_rate'].mean():.2f} bbl/day")
    
    print("\n✓ Datasets saved to data/synthetic/")
    print("  - well_properties.csv")
    print("  - production_data.csv")
    print("  - full_dataset.csv")


if __name__ == "__main__":
    main()
