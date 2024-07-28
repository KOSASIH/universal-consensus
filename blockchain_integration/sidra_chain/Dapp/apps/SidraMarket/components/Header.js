import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { FaShoppingCart } from 'react-icons/fa';
import { AiOutlineMenu } from 'react-icons/ai';
import { useSelector } from 'react-redux';

const Header = () => {
  const [showMenu, setShowMenu] = useState(false);
  const cartItems = useSelector((state) => state.cart.items);

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 768) {
        setShowMenu(false);
      }
    };
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  const toggleMenu = () => {
    setShowMenu(!showMenu);
  };

  return (
    <header className="header">
      <div className="header__container">
        <Link to="/" className="header__logo">
          <img src="logo.png" alt="Logo" />
        </Link>
        <nav className="header__nav">
          <ul>
            <li>
              <Link to="/products">Products</Link>
            </li>
            <li>
              <Link to="/orders">Orders</Link>
            </li>
            <li>
              <Link to="/account">Account</Link>
            </li>
          </ul>
        </nav>
        <div className="header__cart">
          <FaShoppingCart />
          <span className="header__cart-count">{cartItems.length}</span>
        </div>
        <button className="header__menu-btn" onClick={toggleMenu}>
          <AiOutlineMenu />
        </button>
        {showMenu && (
          <div className="header__menu">
            <ul>
              <li>
                <Link to="/products">Products</Link>
              </li>
              <li>
                <Link to="/orders">Orders</Link>
              </li>
              <li>
                <Link to="/account">Account</Link>
              </li>
            </ul>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;
