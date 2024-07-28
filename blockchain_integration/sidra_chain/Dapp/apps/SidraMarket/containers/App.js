import React, { useState, useEffect } from 'react';
import { BrowserRouter, Route, Switch } from 'react-router-dom';
import { Provider } from 'react-redux';
import store from '../store';
import Header from '../components/Header';
import Footer from '../components/Footer';
import UserPage from './UserPage';
import ProductPage from './ProductPage';
import OrderPage from './OrderPage';

const App = () => {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        await store.dispatch(fetchProducts());
        await store.dispatch(fetchOrders());
        setLoading(false);
      } catch (error) {
        console.error(error);
      }
    };
    fetchInitialData();
  }, []);

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <Provider store={store}>
      <BrowserRouter>
        <Header />
        <Switch>
          <Route path="/" exact component={ProductPage} />
          <Route path="/user" component={UserPage} />
          <Route path="/orders" component={OrderPage} />
        </Switch>
        <Footer />
      </BrowserRouter>
    </Provider>
  );
};

export default App;
