import React from 'react';
import { Link } from 'react-router-dom';

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer__container">
        <div className="footer__row">
          <div className="footer__col">
            <h3>About Us</h3>
            <p>
              Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed sit
              amet nulla auctor, vestibulum magna sed, convallis ex.
            </p>
          </div>
          <div className="footer__col">
            <h3>Quick Links</h3>
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
          <div className="footer__col">
            <h3>Contact Us</h3>
            <p>
              <a href="mailto:info@example.com">info@example.com</a>
            </p>
            <p>
              <a href="tel:+1234567890">+1234567890</a>
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
