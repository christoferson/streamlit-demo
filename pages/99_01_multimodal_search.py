import streamlit as st
from PIL import Image
import io
from cmn.multimodal_search import MultimodalSearchService, config
from typing import Union, Tuple

# Page configuration
st.set_page_config(
    page_title="Multimodal Product Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .search-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
    .status-connected {
        color: #28a745;
        font-weight: bold;
    }
    .status-disconnected {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def validate_image(uploaded_file) -> Tuple[bool, str]:
    """Validate uploaded image file"""
    if uploaded_file is None:
        return False, "No file uploaded"

    if uploaded_file.size > config.MAX_FILE_SIZE_MB * 1024 * 1024:
        return False, f"File size exceeds {config.MAX_FILE_SIZE_MB}MB limit"

    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in config.SUPPORTED_IMAGE_FORMATS:
        return False, f"Unsupported format. Use: {', '.join(config.SUPPORTED_IMAGE_FORMATS)}"

    return True, "Valid"

# Header
st.markdown('<h1 class="main-header">🔍 Multimodal Product Search</h1>', unsafe_allow_html=True)

# Initialize service
if 'search_service' not in st.session_state:
    st.session_state.search_service = MultimodalSearchService()

service = st.session_state.search_service

# Sidebar - Connection & Configuration
with st.sidebar:
    st.header("🔌 Connection")

    # Get connection status
    status = service.get_connection_status()

    # Display connection status
    if status['aws_connected']:
        st.markdown('<p class="status-connected">✅ AWS Connected</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-disconnected">❌ AWS Disconnected</p>', unsafe_allow_html=True)

    if status['index_exists']:
        st.markdown('<p class="status-connected">✅ Index Ready</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-disconnected">❌ Index Missing</p>', unsafe_allow_html=True)

    # Connection buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔌 Connect", use_container_width=True):
            with st.spinner("Connecting to AWS..."):
                if service.connect_aws():
                    st.rerun()

    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    st.divider()

    # Index management
    st.header("📊 Index Management")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Create Index", use_container_width=True, disabled=not status['aws_connected']):
            with st.spinner("Creating index..."):
                if service.create_index():
                    st.rerun()

    with col2:
        if st.button("🗑️ Delete Index", use_container_width=True, disabled=not status['aws_connected']):
            if st.session_state.get('confirm_delete', False):
                with st.spinner("Deleting index..."):
                    if service.delete_index():
                        st.session_state.confirm_delete = False
                        st.rerun()
            else:
                st.session_state.confirm_delete = True
                st.warning("Click again to confirm deletion!")
                st.rerun()

    st.divider()

    # Configuration
    st.header("⚙️ Configuration")
    st.info(f"**AWS Profile:** {config.AWS_PROFILE}")
    st.info(f"**Region:** {config.AWS_REGION}")
    st.info(f"**Collection:** {config.COLLECTION_NAME}")
    st.info(f"**Index:** {config.INDEX_NAME}")
    st.info(f"**Endpoint:** {config.OPENSEARCH_ENDPOINT}")

    st.divider()

    # Add this to your sidebar debug tools
    if st.button("🔍 Debug Vector Search"):
        test_query = st.text_input("Enter test query for vector debug", value="METASPEED")
        if test_query:
            debug_info = service.debug_vector_search(test_query)
            st.json(debug_info)

    # Add this to your sidebar debug section
    if st.button("🖼️ Debug Image Search"):
        debug_image_file = st.file_uploader(
            "Upload image for debug", 
            type=['jpg', 'jpeg', 'png', 'webp'], 
            key="debug_image"
        )
        if debug_image_file:
            debug_image = Image.open(debug_image_file)
            debug_info = service.debug_image_search(debug_image)
            st.json(debug_info)


    # Statistics
    st.header("📈 Statistics")
    if status['can_operate']:
        stats = service.get_stats()
        if 'error' not in stats:
            st.metric("Total Products", stats['total_products'])
        else:
            st.error(f"Stats error: {stats['error']}")
    else:
        st.info("Connect and create index to see stats")

    # Settings
    st.header("🔧 Settings")
    search_limit = st.slider("Search Results Limit", 1, 50, 10)

# Main content - only show if properly connected
if not status['can_operate']:
    st.warning("⚠️ Please connect to AWS and create an index to use the application.")
    st.info("Use the sidebar controls to:")
    st.info("1. Click '🔌 Connect' to connect to AWS services")
    st.info("2. Click '➕ Create Index' to create the search index")
else:
    # Main content tabs - Added "Search by Title" tab
    #tab1, tab2, tab3, tab4 = st.tabs(["📝 Register Product", "🖼️ Search by Image", "📝 Search by Text", "🏷️ Search by Title"])
    #tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Register Product", "🖼️ Search by Image", "📝 Search by Text", "🏷️ Search by Title", "📋 List All Products"])
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📝 Register Product", "🖼️ Search by Image", "📝 Search by Text", "🏷️ Search by Title", "📋 List All Products", "✏️ Update Product"])

    # Tab 1: Register Product
    with tab1:
        st.header("📝 Register New Product")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Product Image")
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=config.SUPPORTED_IMAGE_FORMATS,
                help=f"Max size: {config.MAX_FILE_SIZE_MB}MB"
            )

            if uploaded_file:
                is_valid, message = validate_image(uploaded_file)
                if is_valid:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Product Image", use_container_width=True)
                else:
                    st.error(f"❌ {message}")
                    image = None
            else:
                image = None

        with col2:
            st.subheader("Product Details")
            title = st.text_input("Product Title", placeholder="Enter product title...")
            description = st.text_area(
                "Product Description", 
                placeholder="Enter detailed product description...",
                height=200
            )

            if st.button("🚀 Register Product", type="primary", use_container_width=True):
                if image and title and description:
                    success = service.register_product(image, title, description)
                    if success:
                        st.balloons()
                else:
                    st.warning("⚠️ Please provide image, title, and description")

    # Tab 2: Search by Image
    with tab2:
        st.header("🖼️ Search by Image")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Upload Search Image")
            search_image_file = st.file_uploader(
                "Choose search image",
                type=config.SUPPORTED_IMAGE_FORMATS,
                key="search_image",
                help="Upload an image to find similar products"
            )

            if search_image_file:
                is_valid, message = validate_image(search_image_file)
                if is_valid:
                    search_image = Image.open(search_image_file)
                    st.image(search_image, caption="Search Image", use_container_width=True)

                    if st.button("🔍 Search Similar Images", type="primary", use_container_width=True):
                        results = service.search_by_image(search_image, search_limit)
                        st.session_state.image_search_results = results
                else:
                    st.error(f"❌ {message}")

        with col2:
            st.subheader("Search Results")
            if 'image_search_results' in st.session_state:
                results = st.session_state.image_search_results
                if results:
                    st.success(f"Found {len(results)} similar products")
                    for i, result in enumerate(results):
                        with st.container():
                            st.markdown(f"""
                            <div class="result-card">
                                <h4>🏷️ {result['title']}</h4>
                            </div>
                            """, unsafe_allow_html=True)

                            # Add image and details in columns
                            col_img, col_details = st.columns([1, 2])

                            # Column 1: Product Image
                            with col_img:
                                trade_code = result.get('trade_code', '')
                                st.write(f"**Trade Code:** {trade_code}")
                                if trade_code:
                                    image_url = service.generate_image_url_from_trade_code(trade_code)
                                    if image_url:
                                        try:
                                            st.image(image_url, caption=f"Trade Code: {trade_code}", use_container_width=True)
                                        except Exception as e:
                                            st.write("🖼️ Image not available")
                                            st.write(f"**Trade Code:** {trade_code}")
                                    else:
                                        st.write("🖼️ No image URL")
                                        st.write(f"**Trade Code:** {trade_code}")
                                else:
                                    st.write("📷 No trade code")
                                    st.write("No image available")

                            # Column 2: Product Details
                            with col_details:
                                st.write(f"**Score:** {result['score']:.4f}")
                                st.write(f"**Description:** {result['description']}")
                                st.write(f"**ID:** `{result['product_id']}`")

                                # Show trade code and image URL info
                                if trade_code:
                                    st.write(f"**Trade Code:** {trade_code}")
                                    image_url = service.generate_image_url_from_trade_code(trade_code)
                                    if image_url:
                                        st.markdown(f"[🔗 View Image]({image_url})")
                else:
                    st.info("No similar products found")

    # Tab 3: Search by Text
    with tab3:
        st.header("📝 Search by Text")

        search_query = st.text_input(
            "Enter search query",
            placeholder="Describe what you're looking for...",
            help="Use natural language to describe the product you want to find"
        )

        col1, col2, col3 = st.columns([1, 1, 2])
        with col2:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)

        if search_button and search_query:
            results = service.search_by_text(search_query, search_limit)

            if results:
                st.success(f"Found {len(results)} matching products")

                # Display results in a nice format
                for i, result in enumerate(results):
                    with st.expander(f"🏷️ {result['title']} (Score: {result['score']:.4f})"):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.write(f"**Description:** {result['description']}")

                            # Safe access to trade_code
                            trade_code = result.get('trade_code', '')
                            if trade_code:
                                st.write(f"**Trade Code:** {trade_code}")
                                # Generate and show image URL if trade_code exists
                                image_url = service.generate_image_url_from_trade_code(trade_code)
                                if image_url:
                                    st.markdown(f"[🔗 View Image]({image_url})")
                        with col2:
                            st.code(result['product_id'])
            else:
                st.info("No matching products found. Try different keywords.")
        elif search_button:
            st.warning("⚠️ Please enter a search query")

    # Tab 4: Search by Title (NEW)
    with tab4:
        st.header("🏷️ Search by Title")

        col1, col2 = st.columns([2, 1])

        with col1:
            title_query = st.text_input(
                "Enter product title to search",
                placeholder="Enter exact or partial product title...",
                help="Search for products by their title using exact or fuzzy matching"
            )

        with col2:
            st.write("")  # Empty space for alignment
            st.write("")  # Empty space for alignment
            search_type = st.selectbox(
                "Search Type",
                ["Partial Match", "Exact Match", "Fuzzy Match"],
                help="Choose how to match the title"
            )

        col1, col2, col3 = st.columns([1, 1, 2])
        with col2:
            title_search_button = st.button("🔍 Search by Title", type="primary", use_container_width=True)

        if title_search_button and title_query:
            # Map search type to method parameter
            search_mode = {
                "Exact Match": "exact",
                "Fuzzy Match": "fuzzy", 
                "Partial Match": "partial"
            }[search_type]

            results = service.search_by_title(title_query, search_limit, search_mode)

            if results:
                st.success(f"Found {len(results)} products matching '{title_query}'")

                # Display results in a nice format
                for i, result in enumerate(results):
                    with st.expander(f"🏷️ {result['title']} (Score: {result['score']:.4f})"):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.write(f"**Description:** {result['description']}")
                            if 'created_at' in result:
                                st.write(f"**Created:** {result['created_at']}")
                        with col2:
                            st.code(result['product_id'])
            else:
                st.info(f"No products found with title matching '{title_query}'. Try different keywords or search type.")
        elif title_search_button:
            st.warning("⚠️ Please enter a title to search")

    with tab5:
        st.header("📋 List All Products")

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            list_limit = st.number_input("Number of products to show", min_value=1, max_value=100, value=20)

        with col2:
            st.write("")  # Empty space for alignment
            st.write("")  # Empty space for alignment
            if st.button("📋 Load All Products", type="primary", use_container_width=True):
                st.session_state.all_products = service.list_all_products(list_limit)

        # Display products if loaded
        # Tab 5: List All Products - Update the display section
        # Tab 5: List All Products - Update the display section with image display
        if 'all_products' in st.session_state and st.session_state.all_products:
            products = st.session_state.all_products
            st.success(f"Found {len(products)} products")

            # Create a searchable/filterable list
            search_filter = st.text_input("🔍 Filter products by title or description", placeholder="Type to filter...")

            # Filter products if search term provided
            if search_filter:
                filtered_products = [
                    p for p in products 
                    if search_filter.lower() in p['title'].lower() or 
                    search_filter.lower() in p['description'].lower()
                ]
            else:
                filtered_products = products

            if filtered_products:
                st.info(f"Showing {len(filtered_products)} products")

                # Display products in a table-like format
                for i, product in enumerate(filtered_products):
                    with st.expander(f"🏷️ {product['title']}", expanded=False):
                        # Create 3 columns: image, details, actions
                        col1, col2, col3 = st.columns([1, 2, 1])

                        # Column 1: Product Image
                        with col1:
                            trade_code = product.get('trade_code', '')
                            if trade_code:
                                image_url = product.get('image_url', '')
                                if image_url:
                                    try:
                                        # Display the image from URL
                                        st.image(image_url, caption=f"Trade Code: {trade_code}", use_container_width=True)
                                    except Exception as e:
                                        # If image fails to load, show placeholder
                                        st.write("🖼️ Image not available")
                                        st.write(f"**Trade Code:** {trade_code}")
                                else:
                                    st.write("🖼️ No image URL")
                                    st.write(f"**Trade Code:** {trade_code}")
                            else:
                                st.write("📷 No trade code")
                                st.write("No image available")

                        # Column 2: Product Details
                        with col2:
                            st.write(f"**Description:** {product['description']}")
                            st.write(f"**Created:** {product['created_at']}")

                            # Show trade code and image URL info
                            trade_code = product.get('trade_code', '')
                            if trade_code:
                                st.write(f"**Trade Code:** {trade_code}")
                                image_url = product.get('image_url', '')
                                if image_url:
                                    st.write(f"**Image URL:** {image_url}")
                                    # Make it clickable
                                    st.markdown(f"[🔗 Open in New Tab]({image_url})")
                            else:
                                st.write("**Trade Code:** Not set")

                        # Column 3: Actions
                        with col3:
                            st.code(product['product_id'])

                            # Add action buttons
                            if st.button(f"🗑️ Delete", key=f"delete_{product['product_id']}", help="Delete this product"):
                                if service.delete_product(product['product_id']):
                                    st.success("Product deleted!")
                                    # Refresh the list
                                    st.session_state.all_products = service.list_all_products(list_limit)
                                    st.rerun()

                            # Add update button for quick access
                            if st.button(f"✏️ Update", key=f"update_{product['product_id']}", help="Update this product"):
                                # Set the product for update and switch to update tab
                                st.session_state.product_to_update = product
                                st.session_state.switch_to_update = True
                                st.info("💡 Switch to 'Update Product' tab to edit this product")

            else:
                st.info("No products match your filter.")

        elif 'all_products' in st.session_state:
            st.info("No products found in the database.")

    # Tab 6: Update Product (NEW)
    with tab6:
        st.header("✏️ Update Product")

        # Step 1: Find product to update
        st.subheader("1. Find Product to Update")

        col1, col2 = st.columns([2, 1])

        with col1:
            update_product_id = st.text_input(
                "Enter Product ID",
                placeholder="Enter the product ID to update...",
                help="You can find product IDs in the 'List All Products' tab"
            )

        with col2:
            st.write("")  # Empty space for alignment
            st.write("")  # Empty space for alignment
            if st.button("🔍 Find Product", use_container_width=True):
                if update_product_id:
                    found_product = service.get_product(update_product_id)
                    if found_product:
                        st.session_state.product_to_update = found_product
                        st.success(f"✅ Found product: {found_product['title']}")
                    else:
                        st.error("❌ Product not found")
                        if 'product_to_update' in st.session_state:
                            del st.session_state.product_to_update
                else:
                    st.warning("⚠️ Please enter a product ID")

        # Step 2: Show current product and update form
        # In Tab 6: Update Product - Step 2: Show current product information
        if 'product_to_update' in st.session_state:
            current_product = st.session_state.product_to_update

            st.divider()
            st.subheader("2. Current Product Information")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.info(f"**Title:** {current_product['title']}")
                st.info(f"**Product ID:** {current_product['product_id']}")
                st.info(f"**Created:** {current_product['created_at']}")

                # Check if trade_code exists
                trade_code = current_product.get('trade_code', '')
                if trade_code:
                    st.info(f"**Trade Code:** {trade_code}")
                else:
                    st.info("**Trade Code:** Not set")

            with col2:
                with st.expander("Current Description", expanded=False):
                    st.write(current_product['description'])

            st.divider()
            st.subheader("3. Update Product")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("New Product Image (Optional)")
                new_image_file = st.file_uploader(
                    "Choose new image file (leave empty to keep current)",
                    type=config.SUPPORTED_IMAGE_FORMATS,
                    help=f"Max size: {config.MAX_FILE_SIZE_MB}MB",
                    key="update_image"
                )

                if new_image_file:
                    is_valid, message = validate_image(new_image_file)
                    if is_valid:
                        new_image = Image.open(new_image_file)
                        st.image(new_image, caption="New Product Image", use_container_width=True)
                    else:
                        st.error(f"❌ {message}")
                        new_image = None
                else:
                    new_image = None
                    st.info("📷 No new image selected - will keep current image")

            with col2:
                st.subheader("New Product Details")

                new_title = st.text_input(
                    "New Product Title (leave empty to keep current)",
                    placeholder=current_product['title'],
                    help="Leave empty to keep the current title"
                )

                new_description = st.text_area(
                    "New Product Description (leave empty to keep current)",
                    placeholder="Enter new description or leave empty to keep current...",
                    height=150,
                    help="Leave empty to keep the current description"
                )

                # Add trade code input
                new_trade_code = st.text_input(
                    "Trade Code (leave empty to keep current)",
                    placeholder="e.g., 1203A750.020",
                    help="Enter the product trade code"
                )

                # Show current trade code if it exists
                if 'trade_code' in current_product and current_product['trade_code']:
                    st.info(f"Current trade code: {current_product['trade_code']}")

                # Update options
                st.subheader("Update Options")

                col_a, col_b = st.columns(2)

                with col_a:
                    update_title_only = st.checkbox("Update title only")
                    update_description_only = st.checkbox("Update description only")
                    update_trade_code_only = st.checkbox("Update trade code only")

                with col_b:
                    update_image_only = st.checkbox("Update image only")
                    update_all = st.checkbox("Update all fields")

                # Update button
                st.divider()

                if st.button("🔄 Update Product", type="primary", use_container_width=True):
                    # Determine what to update based on checkboxes and inputs
                    title_to_update = None
                    description_to_update = None
                    image_to_update = None
                    trade_code_to_update = None

                    if update_all or not any([update_title_only, update_description_only, update_image_only, update_trade_code_only]):
                        # Update all provided fields
                        if new_title.strip():
                            title_to_update = new_title.strip()
                        if new_description.strip():
                            description_to_update = new_description.strip()
                        if new_trade_code.strip():
                            trade_code_to_update = new_trade_code.strip()
                        if new_image:
                            image_to_update = new_image
                    else:
                        # Update specific fields only
                        if update_title_only and new_title.strip():
                            title_to_update = new_title.strip()
                        if update_description_only and new_description.strip():
                            description_to_update = new_description.strip()
                        if update_trade_code_only and new_trade_code.strip():
                            trade_code_to_update = new_trade_code.strip()
                        if update_image_only and new_image:
                            image_to_update = new_image

                    # Check if anything to update
                    if not any([title_to_update, description_to_update, image_to_update, trade_code_to_update]):
                        st.warning("⚠️ Please provide at least one field to update")
                    else:
                        # Perform update
                        success = service.update_product(
                            current_product['product_id'],
                            image=image_to_update,
                            title=title_to_update,
                            description=description_to_update,
                            trade_code=trade_code_to_update
                        )

                        if success:
                            st.balloons()
                            # Refresh the product info
                            updated_product = service.get_product(current_product['product_id'])
                            if updated_product:
                                st.session_state.product_to_update = updated_product
                                st.rerun()

                # Clear selection button
                if st.button("🗑️ Clear Selection", use_container_width=True):
                    if 'product_to_update' in st.session_state:
                        del st.session_state.product_to_update
                    st.rerun()

        else:
            st.info("👆 Enter a Product ID above to start updating a product")

            # Show recent products for easy selection
            st.subheader("Recent Products")
            recent_products = service.list_all_products(5)

            if recent_products:
                st.write("Click on a Product ID to copy it:")
                for product in recent_products:
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write(f"**{product['title']}**")
                    with col2:
                        st.code(product['product_id'])
                    with col3:
                        if st.button("📋", key=f"copy_{product['product_id']}", help="Click to select this product"):
                            st.session_state.selected_product_id = product['product_id']
                            # Auto-fill the product ID field
                            st.rerun()
            else:
                st.info("No products found. Register some products first.")