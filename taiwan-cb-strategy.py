class TaiwanCBStrategy:
    def __init__(self):
        self.pricer = None
        
    def prepare_data(self, df):
        """
        準備分析所需的資料
        """
        # 整理必要欄位並計算關鍵指標
        analysis_data = pd.DataFrame({
            'date': pd.to_datetime(df['年月'].astype(str), format='%Y%m'),
            'stock_price': df['轉換標的-收盤價'].astype(float),
            'cb_price': df['CB-收盤價'].astype(float),
            'conversion_price': df['轉換價格'].astype(float),
            'conversion_value': df['轉換價值 (1000)'].astype(float),
            'market_premium': df['CB-市值溢價%'].astype(float),
            'remaining_years': (pd.to_datetime(df['到期日']) - pd.to_datetime(df['年月'].astype(str), format='%Y%m')) / np.timedelta64(1, 'Y'),
            'is_guaranteed': (df['是否擔保(Y/N)'] == 'Y').astype(int)
        })
        
        # 計算隱含波動率
        analysis_data['implied_vol'] = analysis_data.apply(
            lambda x: self.calculate_implied_volatility(
                S=x['stock_price'],
                K=x['conversion_price'],
                T=x['remaining_years'],
                r=0.015,  # 假設無風險利率1.5%
                cb_price=x['cb_price']
            ), axis=1
        )
        
        return analysis_data
    
    def calculate_implied_volatility(self, S, K, T, r, cb_price, tolerance=0.0001, max_iter=100):
        """使用牛頓法計算隱含波動率"""
        sigma = 0.3  # 初始猜測值
        
        for i in range(max_iter):
            self.pricer = ConvertibleBondPricer(
                S=S, 
                K=K, 
                T=T, 
                r=r, 
                sigma=sigma,
                conversion_ratio=1.0,
                credit_spread=0.02,
                trigger_price=K*1.3  # 假設轉換價格130%為強制贖回價
            )
            
            price = self.pricer.black_scholes_price()
            diff = price - cb_price
            
            if abs(diff) < tolerance:
                return sigma
                
            # 計算vega
            sigma_up = sigma * 1.01
            self.pricer.sigma = sigma_up
            price_up = self.pricer.black_scholes_price()
            vega = (price_up - price) / (sigma_up - sigma)
            
            if vega == 0:
                return sigma
                
            sigma = sigma - diff / vega
            
            if sigma <= 0:
                return 0.001
                
        return sigma

    def evaluate_pricing_models(self, analysis_data):
        """評估不同定價模型的表現"""
        models_evaluation = {}
        
        # 對每個時點進行模型定價
        for index, row in analysis_data.iterrows():
            self.pricer = ConvertibleBondPricer(
                S=row['stock_price'],
                K=row['conversion_price'],
                T=row['remaining_years'],
                r=0.015,
                sigma=row['implied_vol'],
                conversion_ratio=1.0,
                credit_spread=0.02,
                trigger_price=row['conversion_price']*1.3
            )
            
            # 計算各模型價格
            models_evaluation[index] = {
                'BS': self.pricer.black_scholes_price(),
                'Binomial': self.pricer.binomial_tree_price(),
                'MonteCarlo': self.pricer.monte_carlo_price(),
                'JumpDiffusion': self.pricer.jump_diffusion_price(
                    lambda_jump=2.0,  # 假設平均每年發生2次跳躍
                    mu_jump=-0.1,     # 跳躍平均大小為-10%
                    sigma_jump=0.2    # 跳躍波動率20%
                )
            }
            
        model_prices = pd.DataFrame(models_evaluation).T
        
        # 計算各模型的套利機會
        trading_opportunities = pd.DataFrame()
        for model in ['BS', 'Binomial', 'MonteCarlo', 'JumpDiffusion']:
            # 計算模型價格與市場價格的差異
            mispricing = model_prices[model] - analysis_data['cb_price']
            
            # 計算z-score (標準化的價差)
            z_score = (mispricing - mispricing.rolling(20).mean()) / mispricing.rolling(20).std()
            
            # 生成交易信號
            trading_opportunities[f'{model}_signal'] = np.where(
                z_score > 1.5, -1,  # 賣出信號
                np.where(z_score < -1.5, 1, 0)  # 買入信號
            )
            
            # 計算策略績效
            position = trading_opportunities[f'{model}_signal'].shift(1)
            returns = (analysis_data['cb_price'].pct_change() * position).fillna(0)
            
            trading_opportunities[f'{model}_returns'] = returns
            trading_opportunities[f'{model}_cumulative_returns'] = (1 + returns).cumprod()
            
        return trading_opportunities
    
    def plot_results(self, analysis_data, trading_opportunities):
        """繪製分析結果"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 20))
        
        # 1. 價格和隱含波動率
        ax1 = axes[0]
        ax1.plot(analysis_data.index, analysis_data['cb_price'], 'b-', label='市場價格')
        ax1.set_ylabel('價格')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(analysis_data.index, analysis_data['implied_vol'], 'r--', label='隱含波動率')
        ax1_twin.set_ylabel('隱含波動率')
        ax1.set_title('CB價格和隱含波動率')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2)
        
        # 2. 各模型交易信號
        ax2 = axes[1]
        for model in ['BS', 'Binomial', 'MonteCarlo', 'JumpDiffusion']:
            signals = trading_opportunities[f'{model}_signal']
            buy_points = signals[signals == 1].index
            sell_points = signals[signals == -1].index
            
            ax2.scatter(buy_points, analysis_data.loc[buy_points, 'cb_price'], 
                       marker='^', label=f'{model} Buy')
            ax2.scatter(sell_points, analysis_data.loc[sell_points, 'cb_price'], 
                       marker='v', label=f'{model} Sell')
        
        ax2.plot(analysis_data.index, analysis_data['cb_price'], 'k-', alpha=0.5)
        ax2.set_title('交易信號')
        ax2.legend()
        
        # 3. 績效比較
        ax3 = axes[2]
        for model in ['BS', 'Binomial', 'MonteCarlo', 'JumpDiffusion']:
            ax3.plot(trading_opportunities.index, 
                    trading_opportunities[f'{model}_cumulative_returns'],
                    label=f'{model}')
        ax3.set_title('累積報酬率')
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
